#pragma once
#include <string>
#include <vector>

#include <opencv2/core.hpp>

#include "place_rec/types.hpp"

namespace pr{

struct TumFrame {

    double stamp;
    std::string rgb_path;
    std::string depth_path;
    
};

class TumSequence {

public:
    explicit TumSequence(const std::string& root, double max_dt = 0.02);
    std::vector<TumFrame> keyframes (double period) const;

    static cv::Mat load_gray(const std::string& path);
    static cv::Mat load_depth_m(const std::string& path);
    // TUM recommendation for the pre-registered depth: ROS defaults, no undistorsion.
    static CameraIntrinsics intrinsics() {return {525.0, 525.0, 319.5, 239.5};}

private:
    std::vector<TumFrame> frames_;


};

} // namespace pr