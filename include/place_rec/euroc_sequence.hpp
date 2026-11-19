#pragma once
#include <opencv2/core.hpp>
#include <string>
#include <vector>

// #include "place_rec/types.hpp"

namespace pr {

struct EurocFrame {
  double stamp;
  std::string gray_path;
};

class EurocSequence {
 public:
  explicit EurocSequence(const std::string& root);
  std::vector<EurocFrame> keyframes(double period) const;
  std::size_t size() const { return frames_.size(); }
  static cv::Mat load_gray(const std::string& path);
  // sensor.yaml contains the distornions
  // will be defined later
  //   static CameraIntrinsics intrinsics();

 private:
  std::vector<EurocFrame> frames_;
};

}  // namespace pr