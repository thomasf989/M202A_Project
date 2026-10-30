#include "../Native/LocalizerCore.hpp"
#include "../Native/RoomBuilder.hpp"
#include <opencv2/calib3d.hpp>
#include <fstream>
#include <iostream>
#include <iterator>
#include <stdexcept>

static void geometryTest() {
    cv::RNG rng(42); std::vector<cv::Point3d> world; std::vector<int> ids, groups;
    for (int i = 0; i < 120; ++i) { world.emplace_back(rng.uniform(-2., 2.), rng.uniform(-1.5, 1.5), rng.uniform(3., 7.)); ids.push_back(i); groups.push_back(i % 3); }
    cv::Matx33d K(600, 0, 320, 0, 600, 240, 0, 0, 1);
    cv::Mat r = (cv::Mat_<double>(3, 1) << .1, .08, -.03), t = (cv::Mat_<double>(3, 1) << .1, -.2, .3);
    std::vector<cv::Point2d> pixels; cv::projectPoints(world, r, t, K, cv::noArray(), pixels);
    for (int i = 0; i < 36; ++i) pixels[i] = cv::Point2d(rng.uniform(0., 640.), rng.uniform(0., 480.));
    auto solution = roomloc::solvePose(world, pixels, ids, groups, K, {640, 480});
    cv::Mat R; cv::Rodrigues(r, R); cv::Mat truth = -R.t() * t;
    cv::Vec3d estimated(solution.cameraToWorld(0, 3), solution.cameraToWorld(1, 3), solution.cameraToWorld(2, 3));
    if (!solution.success || cv::norm(estimated - cv::Vec3d(truth.at<double>(0), truth.at<double>(1), truth.at<double>(2))) > .02)
        throw std::runtime_error("Pose recovery with 30 percent outliers failed");
    world.resize(10); cv::projectPoints(world, r, t, K, cv::noArray(), pixels);
    auto originals = world; auto observations = pixels;
    world.clear(); pixels.clear(); ids.clear(); groups.clear();
    for (int g = 0; g < 3; ++g) for (int i = 0; i < 10; ++i) {
        world.push_back(originals[i]); pixels.push_back(observations[i]); ids.push_back(i); groups.push_back(g);
    }
    auto duplicate = roomloc::solvePose(world, pixels, ids, groups, K, {640, 480});
    if (duplicate.success || duplicate.stats.inliers > 10) throw std::runtime_error("Repeated query features counted more than once");
    std::cout << "Native geometry and unique-vote checks passed\n";
}

int main(int argc, char** argv) {
    try {
        if (argc == 2 && std::string(argv[1]) == "--geometry-selftest") { geometryTest(); return 0; }
        if (argc == 3 && std::string(argv[1]) == "--inspect") { std::cout << roomloc::Localizer(argv[2]).metadataJSON(); return 0; }
        if (argc == 7 && (std::string(argv[1]) == "--build-room" || std::string(argv[1]) == "--cancel-build")) {
            roomloc::BuildOptions options; options.name = argv[5]; options.maxReferences = std::stoi(argv[6]);
            bool cancel = std::string(argv[1]) == "--cancel-build";
            roomloc::buildRoom(argv[2], argv[3], argv[4], options, [cancel](double fraction, const std::string&) { return !cancel || fraction < .5; });
            std::cout << roomloc::Localizer(std::string(argv[4]) + "/map.yml").metadataJSON(); return 0;
        }
        if (argc != 4) { std::cerr << "Usage: roomloc_native map.yml query.bgr camera.json\n"; return 1; }
        roomloc::Localizer engine(argv[1]);
        std::ifstream raw(argv[2], std::ios::binary); uint32_t dimensions[2] = {};
        raw.read(reinterpret_cast<char*>(dimensions), 8);
        if (!raw || dimensions[0] < 32 || dimensions[1] < 32 || dimensions[0] > 1280 || dimensions[1] > 1280)
            throw std::runtime_error("Invalid raw test image");
        cv::Mat image(int(dimensions[1]), int(dimensions[0]), CV_8UC3);
        raw.read(reinterpret_cast<char*>(image.data), image.total() * 3);
        if (!raw || raw.peek() != std::char_traits<char>::eof()) throw std::runtime_error("Incomplete raw test image");
        std::ifstream request(argv[3]); std::string camera((std::istreambuf_iterator<char>(request)), {});
        std::cout << engine.localize(image, camera);
    } catch (const std::exception& e) { std::cerr << e.what() << '\n'; return 2; }
}
