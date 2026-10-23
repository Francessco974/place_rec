#pragma once
#include <vector>

#include <opencv2/core.hpp>

namespace pr {

struct CameraIntrinsics {
    double fx = 0, fy = 0, cx = 0, cy = 0;
};

struct Keyframe {
    int id = -1;
    double stamp = 0.0;
    cv::Mat gray;
    std::vector<cv::KeyPoint> keypoints;
    
    cv::Mat descriptors;
    std::vector<cv::Point3f> points3d;
};

struct Candidate {
    int id = -1;
    double score = 0.0;
};

} // namespace pr


