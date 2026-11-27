#pragma once

#include <opencv2/calib3d.hpp>
#include <opencv2/core.hpp>
#include <opencv2/core/types.hpp>
#include <vector>

#include "place_rec/matcher.hpp"

namespace pr {

enum class PnpBackend { OpenCv, Own };

struct PnpParams {
  int iterations_count = 100;
  float reprojection_error_px = 8.0f;  // Max allowed pixel error for an inlier
  double confidence = 0.99;
  int opencv_flag = cv::SOLVEPNP_P3P;
  PnpBackend backend{PnpBackend::OpenCv};
};

struct CorrespondencesPnP {
  std::vector<cv::Point3f> object_points{};
  std::vector<cv::Point2f> image_points{};
  std::vector<int> candidate_ids{};
  std::vector<int> query_ids{};
};

struct PnpResult {
  bool success{false};
  //   maps candidate camera frame → query camera frame
  cv::Mat rvec{};
  cv::Mat tvec{};
  //   indices into the correspondence list (not keypoint ids; map them through candidate_ids /
  //   query_ids)
  std::vector<int> inliers{};
};

CorrespondencesPnP gather_input_PnPRansac(const std::vector<cv::Point3f>& candidate,
                                          const std::vector<cv::KeyPoint>& query,
                                          const std::vector<MatchDescription>& matches);

//   K must be 3×3 CV_64F; empty dist_coeffs means no distortion
PnpResult run_PnPRansac(const CorrespondencesPnP& data, const cv::Mat& K,
                        const cv::Mat& dist_coeffs, const PnpParams& params);

}  // namespace pr