#pragma once
#include <opencv2/core/cvstd_wrapper.hpp>
#include <opencv2/core/mat.hpp>
#include <opencv2/features2d.hpp>

#include "place_rec/types.hpp"

namespace pr {
struct OrbParams {
  int n_features = 1000;
  float scale_factor = 1.2f;
  int n_levels = 8;
  int fast_threshold = 20;
};

struct DepthParams {
  float z_min = 0.3f, z_max = 6.0f;
  float edge_tol = 0.05f;
};

class FeatureExtractor {
 public:
  explicit FeatureExtractor(OrbParams p = {});
  void extract(Keyframe& kf) const;
  static void attach_depth(Keyframe& kf, const cv::Mat& depth_m, const CameraIntrinsics& K,
                           const DepthParams& p = {});

 private:
  cv::Ptr<cv::ORB> orb_;
};

}  // namespace pr