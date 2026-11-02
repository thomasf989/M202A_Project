#include "LocalizerCore.hpp"
#include <opencv2/calib3d.hpp>
#include <opencv2/features2d.hpp>
#include <opencv2/imgproc.hpp>
#include <algorithm>
#include <chrono>
#include <cmath>
#include <iomanip>
#include <map>
#include <numeric>
#include <set>
#include <sstream>
#include <stdexcept>

namespace roomloc {
namespace {
std::string quote(const std::string& s) {
    std::ostringstream out; out << '"';
    for (unsigned char c : s) {
        if (c == '"' || c == '\\') out << '\\' << c;
        else if (c < 32) out << "\\u" << std::hex << std::setw(4) << std::setfill('0') << int(c) << std::dec;
        else out << c;
    }
    out << '"'; return out.str();
}
cv::Mat checkedMatrix(const cv::FileNode& node, int rows, int cols, const std::string& dt, int maxRows) {
    if (!node.isMap() || node["rows"].empty() || node["cols"].empty()) throw std::runtime_error("Missing map matrix");
    int r = int(node["rows"]), c = int(node["cols"]);
    if (r < 1 || r > maxRows || c != cols || (rows > 0 && r != rows) || std::string(node["dt"]) != dt)
        throw std::runtime_error("Invalid map matrix dimensions or type");
    cv::Mat value; node >> value;
    if (value.rows != r || value.cols != c || !cv::checkRange(value, true, nullptr, -1e6, 1e6))
        throw std::runtime_error("Invalid or nonfinite map data");
    return value;
}
double median(std::vector<double> values) {
    if (values.empty()) return 0;
    std::sort(values.begin(), values.end()); size_t n = values.size();
    return n % 2 ? values[n / 2] : .5 * (values[n / 2 - 1] + values[n / 2]);
}
std::vector<int> score(const std::vector<cv::Point3d>& world, const std::vector<cv::Point2d>& pixels,
                       const std::vector<int>& qids, const cv::Matx33d& K,
                       const cv::Mat& r, const cv::Mat& t, std::vector<double>& errors) {
    std::vector<cv::Point2d> projected; cv::projectPoints(world, r, t, K, cv::noArray(), projected);
    cv::Mat rotation; cv::Rodrigues(r, rotation);
    std::vector<int> order(world.size()); std::iota(order.begin(), order.end(), 0);
    errors.resize(world.size());
    for (size_t i = 0; i < world.size(); ++i) errors[i] = cv::norm(projected[i] - pixels[i]);
    std::stable_sort(order.begin(), order.end(), [&](int a, int b) { return errors[a] < errors[b]; });
    std::set<int> seen; std::vector<int> kept;
    for (int i : order) {
        double depth = rotation.at<double>(2, 0) * world[i].x + rotation.at<double>(2, 1) * world[i].y +
                       rotation.at<double>(2, 2) * world[i].z + t.at<double>(2);
        if (depth > .05 && errors[i] < 4. && seen.insert(qids[i]).second) kept.push_back(i);
    }
    return kept;
}
double supportMedian(const std::vector<int>& ids, const std::vector<double>& errors) {
    std::vector<double> values; for (int i : ids) values.push_back(errors[i]); return median(values);
}
cv::Mat rootSIFT(const cv::Mat& image, int count, std::vector<cv::KeyPoint>& points) {
    cv::Mat gray, mask, descriptors;
    if (image.channels() == 1) gray = image;
    else cv::cvtColor(image, gray, cv::COLOR_BGR2GRAY);
    cv::compare(gray, 8, mask, cv::CMP_GT);
    cv::erode(mask, mask, cv::Mat::ones(9, 9, CV_8U));
    cv::SIFT::create(count, 3, .025)->detectAndCompute(gray, mask, points, descriptors);
    for (int i = 0; i < descriptors.rows; ++i) {
        float* row = descriptors.ptr<float>(i); double sum = 0;
        for (int j = 0; j < 128; ++j) sum += row[j];
        for (int j = 0; j < 128; ++j) row[j] = std::sqrt(row[j] / std::max(sum, 1e-8));
    }
    return descriptors;
}
struct Cameras { std::vector<cv::Matx33d> candidates; cv::Mat distortion; bool estimated = true; std::string source; };
Cameras cameras(const cv::FileStorage& request, const cv::Mat& image, const std::string& dataset,
                const std::string& datasetCameraJSON) {
    double ow = double(request["original_width"]), oh = double(request["original_height"]);
    if (!std::isfinite(ow) || !std::isfinite(oh) || ow < 32 || oh < 32 || ow * oh > 65e6 ||
        std::abs((ow / oh) / (double(image.cols) / image.rows) - 1.) > .005)
        throw std::runtime_error("Invalid photo size or aspect ratio");
    std::string mode = request["mode"].empty() ? "auto" : std::string(request["mode"]);
    Cameras result; result.distortion = cv::Mat::zeros(1, 5, CV_64F);
    if (mode == "saved" || mode == "dataset") {
        cv::FileStorage stored;
        cv::FileNode profile;
        if (mode == "dataset") {
            if (dataset != "7-scenes" || ow != 640 || oh != 480)
                throw std::runtime_error("Dataset intrinsics require an original 640x480 7-Scenes query");
            stored.open(datasetCameraJSON, cv::FileStorage::READ | cv::FileStorage::MEMORY | cv::FileStorage::FORMAT_JSON);
            profile = stored.root();
        } else profile = request["calibration"];
        for (const char* key : {"width", "height", "fx", "fy", "cx", "cy"})
            if (profile[key].empty() || !std::isfinite(double(profile[key]))) throw std::runtime_error("Invalid calibration profile");
        double pw = double(profile["width"]), ph = double(profile["height"]);
        if (pw < 32 || ph < 32 || double(profile["fx"]) <= 0 || double(profile["fy"]) <= 0 ||
            std::abs((ow / oh) / (pw / ph) - 1.) > .005) throw std::runtime_error("Photo and calibration aspect ratios differ");
        double sx = image.cols / pw, sy = image.rows / ph;
        result.candidates.push_back(cv::Matx33d(double(profile["fx"]) * sx, 0, (double(profile["cx"]) + .5) * sx - .5,
            0, double(profile["fy"]) * sy, (double(profile["cy"]) + .5) * sy - .5, 0, 0, 1));
        auto d = profile["distortion"];
        if (!d.empty()) {
            if (!d.isSeq() || (d.size() != 4 && d.size() != 5 && d.size() != 8 && d.size() != 12 && d.size() != 14))
                throw std::runtime_error("Invalid lens distortion coefficients");
            result.distortion = cv::Mat(1, int(d.size()), CV_64F); int i = 0;
            for (auto value : d) { double v = double(value); if (!std::isfinite(v)) throw std::runtime_error("Invalid distortion"); result.distortion.at<double>(i++) = v; }
        }
        result.estimated = mode == "dataset"; result.source = mode; return result;
    }
    if (mode != "auto") throw std::runtime_error("Unknown camera mode");
    double focal35 = request["focal_35mm"].empty() ? 0 : double(request["focal_35mm"]);
    std::vector<double> focals;
    if (std::isfinite(focal35) && focal35 >= 8 && focal35 <= 500) {
        double f = focal35 * std::hypot(image.cols, image.rows) / std::hypot(36., 24.);
        focals = {.9 * f, f, 1.1 * f}; result.source = "exif";
    } else {
        for (double scale : {.55, .70, .90, 1.15, 1.45}) focals.push_back(scale * std::max(image.cols, image.rows));
        result.source = "search";
    }
    for (double f : focals) result.candidates.push_back(cv::Matx33d(f, 0, (image.cols - 1.) / 2., 0, f, (image.rows - 1.) / 2., 0, 0, 1));
    return result;
}
void matrixJSON(std::ostream& out, const cv::Matx44d& value) {
    out << '['; for (int i = 0; i < 4; ++i) { if (i) out << ','; out << '[';
        for (int j = 0; j < 4; ++j) { if (j) out << ','; out << value(i, j); } out << ']'; } out << ']';
}
}

cv::Mat extractRootSIFT(const cv::Mat& image, int count, std::vector<cv::KeyPoint>& points) {
    return rootSIFT(image, count, points);
}

Solution solvePose(const std::vector<cv::Point3d>& world, const std::vector<cv::Point2d>& pixels,
                   const std::vector<int>& qids, const std::vector<int>& groups,
                   const cv::Matx33d& K, cv::Size imageSize) {
    Solution result;
    if (world.size() != pixels.size() || world.size() != qids.size() || world.size() != groups.size())
        throw std::runtime_error("Mismatched correspondence arrays");
    result.stats.correspondences = int(world.size());
    if (world.size() < 12) return result;
    std::map<int, std::vector<int>> subsets; auto& all = subsets[-1];
    for (int i = 0; i < int(world.size()); ++i) { all.push_back(i); subsets[groups[i]].push_back(i); }
    cv::Mat bestR, bestT; std::vector<int> inliers; std::vector<double> errors;
    double bestMedian = 1e9;
    for (const auto& row : subsets) {
        if (row.second.size() < 12) continue;
        std::vector<cv::Point3d> points; std::vector<cv::Point2d> observed;
        for (int i : row.second) { points.push_back(world[i]); observed.push_back(pixels[i]); }
        cv::Mat r, t, support; cv::setRNGSeed(42);
        bool ok = cv::solvePnPRansac(points, observed, K, cv::noArray(), r, t, false, 3000, 4., .999, support, cv::SOLVEPNP_EPNP);
        if (!ok || support.empty() || !cv::checkRange(r) || !cv::checkRange(t)) continue;
        std::vector<double> candidateErrors; auto ids = score(world, pixels, qids, K, r, t, candidateErrors);
        double m = supportMedian(ids, candidateErrors);
        if (ids.size() > inliers.size() || (ids.size() == inliers.size() && m < bestMedian)) {
            bestR = r.clone(); bestT = t.clone(); inliers = ids; errors = candidateErrors; bestMedian = m;
        }
    }
    result.reason = "pnp_ransac_failed"; if (bestR.empty()) return result;
    for (int step = 0; step < 2 && inliers.size() >= 12; ++step) {
        std::vector<cv::Point3d> points; std::vector<cv::Point2d> observed;
        for (int i : inliers) { points.push_back(world[i]); observed.push_back(pixels[i]); }
        cv::solvePnPRefineLM(points, observed, K, cv::noArray(), bestR, bestT);
        inliers = score(world, pixels, qids, K, bestR, bestT, errors);
    }
    std::set<int> unique(qids.begin(), qids.end()), referenceSupport;
    std::set<std::pair<int, int>> cells;
    for (int i : inliers) {
        cells.emplace(int(std::floor(pixels[i].x * 4 / imageSize.width)), int(std::floor(pixels[i].y * 4 / imageSize.height)));
        referenceSupport.insert(groups[i]);
    }
    result.stats.uniqueMatches = int(unique.size()); result.stats.inliers = int(inliers.size());
    result.stats.inlierRatio = double(inliers.size()) / std::max<size_t>(1, unique.size());
    result.stats.occupiedCells = int(cells.size()); result.stats.references = int(referenceSupport.size());
    result.stats.medianError = supportMedian(inliers, errors);
    result.reason = "insufficient_geometric_support";
    if (inliers.size() < 12 || result.stats.inlierRatio < .12 || cells.size() < 4) return result;
    cv::Mat cloud(int(inliers.size()), 3, CV_64F); cv::Point3d mean;
    for (int i : inliers) mean += world[i];
    mean *= 1. / inliers.size();
    for (int j = 0; j < int(inliers.size()); ++j) {
        const auto p = world[inliers[j]] - mean;
        cloud.at<double>(j, 0) = p.x; cloud.at<double>(j, 1) = p.y; cloud.at<double>(j, 2) = p.z;
    }
    cv::Mat singular; cv::SVD::compute(cloud, singular);
    result.reason = "nearly_collinear_3d_points"; if (singular.at<double>(1) < .03) return result;
    cv::Mat R; cv::Rodrigues(bestR, R); cv::Mat rt = R.t(), center = -rt * bestT;
    for (int i = 0; i < 3; ++i) {
        for (int j = 0; j < 3; ++j) result.cameraToWorld(i, j) = rt.at<double>(i, j);
        result.cameraToWorld(i, 3) = center.at<double>(i);
    }
    result.success = true; result.reason.clear(); return result;
}

Localizer::Localizer(const std::string& path) {
    cv::FileStorage fs(path, cv::FileStorage::READ);
    if (!fs.isOpened() || std::string(fs["format"]) != "roomloc-map-v1" ||
        std::string(fs["units"]) != "meters" || std::string(fs["camera_convention"]) != "opencv-camera-to-world")
        throw std::runtime_error("Unsupported mobile room map");
    name_ = std::string(fs["name"]); format_ = std::string(fs["dataset_format"]);
    datasetCameraJSON_ = std::string(fs["dataset_camera_json"]);
    if (name_.size() > 1024 || datasetCameraJSON_.size() > 16384) throw std::runtime_error("Invalid map metadata");
    nfeatures_ = int(fs["nfeatures"]);
    if (nfeatures_ < 16 || nfeatures_ > 10000) throw std::runtime_error("Unsupported feature count");
    centers_ = checkedMatrix(fs["centers"], -1, 128, "f", 4096);
    idf_ = checkedMatrix(fs["idf"], 1, centers_.rows, "f", 1);
    auto refs = fs["references"];
    if (!refs.isSeq() || refs.size() < 4 || refs.size() > 1000) throw std::runtime_error("Invalid reference count");
    vectors_ = checkedMatrix(fs["vectors"], int(refs.size()), centers_.rows, "f", 1000);
    std::set<std::string> ids; size_t features = 0;
    for (const auto& node : refs) {
        Reference ref; ref.id = std::string(node["id"]); ref.hash = std::string(node["image_sha256"]);
        if (ref.id.empty() || ref.id.size() > 128 || !ids.insert(ref.id).second || (!ref.hash.empty() && ref.hash.size() != 64))
            throw std::runtime_error("Invalid reference identity");
        int rows = int(node["descriptors"]["rows"]);
        if (rows < 2 || rows > 10000 || features + size_t(rows) > 400000) throw std::runtime_error("Reference map is too large for this app");
        ref.descriptors = checkedMatrix(node["descriptors"], rows, 128, "f", 10000);
        ref.world = checkedMatrix(node["world_points"], rows, 3, "f", 10000);
        ref.valid = checkedMatrix(node["valid"], rows, 1, "u", 10000);
        features += rows; references_.push_back(std::move(ref));
    }
}

std::string Localizer::metadataJSON() const {
    std::ostringstream out;
    out << "{\"name\":" << quote(name_) << ",\"references\":" << references_.size()
        << ",\"dataset_format\":" << quote(format_) << ",\"dataset_camera_json\":" << quote(datasetCameraJSON_) << '}';
    return out.str();
}

std::string Localizer::localize(const cv::Mat& input, const std::string& cameraJSON) const {
    auto start = std::chrono::steady_clock::now(); cv::setNumThreads(2);
    if (input.empty() || input.depth() != CV_8U || (input.channels() != 1 && input.channels() != 3) ||
        input.cols < 32 || input.rows < 32 || std::max(input.cols, input.rows) > 1280)
        throw std::runtime_error("Invalid processed photo");
    cv::FileStorage request(cameraJSON, cv::FileStorage::READ | cv::FileStorage::MEMORY | cv::FileStorage::FORMAT_JSON);
    Cameras camera = cameras(request, input, format_, datasetCameraJSON_);
    cv::Mat image = input;
    if (cv::norm(camera.distortion) > 0) cv::undistort(input, image, camera.candidates[0], camera.distortion);
    std::vector<cv::KeyPoint> keypoints; cv::Mat desc = rootSIFT(image, nfeatures_, keypoints);
    Solution best; best.reason = "no_valid_depth_matches"; cv::Matx33d selectedK = camera.candidates.front();
    std::vector<cv::Point3d> world; std::vector<cv::Point2d> pixels; std::vector<int> qids, groups;
    int excluded = 0;
    if (desc.rows >= 2) {
        std::vector<cv::DMatch> assigned; cv::BFMatcher(cv::NORM_L2).match(desc, centers_, assigned);
        cv::Mat vector = cv::Mat::zeros(1, centers_.rows, CV_32F);
        for (const auto& m : assigned) vector.at<float>(m.trainIdx) += 1.f;
        for (int j = 0; j < vector.cols; ++j) vector.at<float>(j) = std::sqrt(vector.at<float>(j) / std::max(1, desc.rows)) * idf_.at<float>(j);
        double length = cv::norm(vector); if (length > 0) vector /= length;
        std::string hash = request["source_sha256"].empty() ? "" : std::string(request["source_sha256"]);
        std::vector<std::pair<double, int>> ranking;
        for (int i = 0; i < int(references_.size()); ++i) {
            if (!hash.empty() && references_[i].hash == hash) { ++excluded; continue; }
            ranking.emplace_back(vectors_.row(i).dot(vector), i);
        }
        std::stable_sort(ranking.begin(), ranking.end(), [](auto a, auto b) { return a.first > b.first; });
        for (int group = 0; group < std::min<int>(8, ranking.size()); ++group) {
            const auto& ref = references_[ranking[group].second];
            cv::FlannBasedMatcher matcher(cv::makePtr<cv::flann::KDTreeIndexParams>(4), cv::makePtr<cv::flann::SearchParams>(64));
            std::vector<std::vector<cv::DMatch>> forward, backward;
            matcher.knnMatch(desc, ref.descriptors, forward, 2); matcher.knnMatch(ref.descriptors, desc, backward, 1);
            for (const auto& pair : forward) {
                if (pair.size() != 2) continue;
                const auto& a = pair[0];
                if (a.distance >= .75f * pair[1].distance || backward[a.trainIdx].empty() ||
                    backward[a.trainIdx][0].trainIdx != a.queryIdx || !ref.valid.at<unsigned char>(a.trainIdx)) continue;
                auto xyz = ref.world.ptr<float>(a.trainIdx);
                world.emplace_back(xyz[0], xyz[1], xyz[2]); pixels.emplace_back(keypoints[a.queryIdx].pt);
                qids.push_back(a.queryIdx); groups.push_back(group);
            }
        }
        for (const auto& K : camera.candidates) {
            auto solution = solvePose(world, pixels, qids, groups, K, image.size());
            if ((!best.success && solution.success) || (best.success && solution.success &&
                (solution.stats.inliers > best.stats.inliers || (solution.stats.inliers == best.stats.inliers && solution.stats.medianError < best.stats.medianError)))) {
                best = solution; selectedK = K;
            } else if (!best.success) best = solution;
        }
    }
    double elapsed = std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
    std::ostringstream out; out << std::setprecision(12);
    out << "{\"success\":" << (best.success ? "true" : "false") << ",\"method\":\"RootSIFT + TF-IDF + PnP/RANSAC (on device)\""
        << ",\"reason\":" << quote(best.reason) << ",\"query_features\":" << keypoints.size()
        << ",\"excluded_reference_count\":" << excluded << ",\"inliers\":" << best.stats.inliers
        << ",\"inlier_ratio\":" << best.stats.inlierRatio << ",\"occupied_grid_cells\":" << best.stats.occupiedCells
        << ",\"supporting_references\":" << best.stats.references << ",\"median_reprojection_px\":" << best.stats.medianError
        << ",\"elapsed_seconds\":" << elapsed << ",\"camera_estimated\":" << (camera.estimated ? "true" : "false")
        << ",\"camera_source\":" << quote(camera.source);
    if (best.success) {
        const auto& T = best.cameraToWorld;
        out << ",\"pose\":{\"T_world_camera_cv\":"; matrixJSON(out, T);
        out << ",\"position_m\":[" << T(0, 3) << ',' << T(1, 3) << ',' << T(2, 3)
            << "],\"forward_world\":[" << T(0, 2) << ',' << T(1, 2) << ',' << T(2, 2) << "]}";
        out << ",\"query_intrinsics\":[";
        for (int i = 0; i < 3; ++i) { if (i) out << ','; out << '[';
            for (int j = 0; j < 3; ++j) { if (j) out << ','; out << selectedK(i, j); } out << ']'; }
        out << ']';
    }
    out << '}'; return out.str();
}
}
