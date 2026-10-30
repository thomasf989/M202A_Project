#pragma once
#include <opencv2/core.hpp>
#include <string>
#include <vector>

namespace roomloc {
cv::Mat extractRootSIFT(const cv::Mat& image, int count, std::vector<cv::KeyPoint>& points);
struct Statistics {
    int correspondences = 0, uniqueMatches = 0, inliers = 0, occupiedCells = 0, references = 0;
    double inlierRatio = 0, medianError = 0;
};
struct Solution {
    bool success = false;
    cv::Matx44d cameraToWorld = cv::Matx44d::eye();
    Statistics stats;
    std::string reason = "too_few_2d_3d_correspondences";
};
// Exposed separately so the geometry used on iPhone can be tested without UIKit.
Solution solvePose(const std::vector<cv::Point3d>& world, const std::vector<cv::Point2d>& pixels,
                   const std::vector<int>& queryIDs, const std::vector<int>& groups,
                   const cv::Matx33d& K, cv::Size imageSize);

class Localizer {
public:
    explicit Localizer(const std::string& mapPath);
    std::string metadataJSON() const;
    // BGR or grayscale pixels must already be EXIF-oriented and <=1280 px.
    // Camera JSON records the ORIGINAL oriented dimensions before resizing.
    std::string localize(const cv::Mat& image, const std::string& cameraJSON) const;
private:
    struct Reference { std::string id, hash; cv::Mat descriptors, world, valid; };
    std::string name_, format_, datasetCameraJSON_;
    int nfeatures_ = 3000;
    cv::Mat centers_, idf_, vectors_;
    std::vector<Reference> references_;
};
}
