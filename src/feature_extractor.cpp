#include "place_rec/feature_extractor.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <opencv2/core/mat.hpp>
#include <opencv2/core/types.hpp>
#include <opencv2/features2d.hpp>

#include "place_rec/types.hpp"

namespace pr {

FeatureExtractor::FeatureExtractor(OrbParams p)
    : orb_(cv::ORB::create(p.n_features, p.scale_factor, p.n_levels,
                           /*edgeThreshold=*/31, /*firstLevel=*/0, /*WTA_K=*/2,
                           cv::ORB::HARRIS_SCORE,
                           /*patchSize=*/31, p.fast_threshold)) {}

void FeatureExtractor::extract(Keyframe& kf) const {
  orb_->detectAndCompute(kf.gray, cv::noArray(), kf.keypoints, kf.descriptors);
}

void FeatureExtractor::attach_depth(Keyframe& kf, const cv::Mat& depth_m, const CameraIntrinsics& K,
                                    const DepthParams& p) {
  CV_Assert(depth_m.type() == CV_32FC1);
  CV_Assert(depth_m.size() == kf.gray.size());
  kf.points3d.assign(kf.keypoints.size(), cv::Point3f(0.f, 0.f, -1.f));
  for (size_t i = 0; i < kf.keypoints.size(); ++i) {
    const int u = static_cast<int>(std::lround(kf.keypoints[i].pt.x));
    const int v = static_cast<int>(std::lround(kf.keypoints[i].pt.y));
    if (u < 1 || v < 1 || u >= depth_m.cols - 1 || v >= depth_m.rows - 1) continue;
    float lo = 1e9f, hi = -1e9f;
    for (int dv = -1; dv <= 1; ++dv)
      for (int du = -1; du <= 1; ++du) {
        const float z = depth_m.at<float>(v + dv, u + du);
        lo = std::min(lo, z);
        hi = std::max(hi, z);
      }
    const float z = depth_m.at<float>(v, u);
    if (lo < p.z_min || hi > p.z_max || hi - lo > p.edge_tol * z) continue;
    const float x = static_cast<float>((kf.keypoints[i].pt.x - K.cx) * z / K.fx);
    const float y = static_cast<float>((kf.keypoints[i].pt.y - K.cy) * z / K.fy);
    kf.points3d[i] = cv::Point3f(x, y, z);
  }
}
}  // namespace pr