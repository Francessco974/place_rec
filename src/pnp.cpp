#include "place_rec/pnp.hpp"

#include <cmath>
#include <opencv2/core/types.hpp>
#include <stdexcept>

namespace pr {

namespace {
PnpResult pnp_opencv(const CorrespondencesPnP& data, const cv::Mat& K, const cv::Mat& dist,
                     const PnpParams& params) {
  PnpResult result{};
  result.success =
      cv::solvePnPRansac(data.object_points, data.image_points, K, dist, result.rvec, result.tvec,
                         false, params.iterations_count, params.reprojection_error_px,
                         params.confidence, result.inliers, params.opencv_flag);

  if (!result.success) {
    result.inliers.clear();
    result.rvec.release();
    result.tvec.release();
  }
  return result;
}
// void pnp_own(...);

}  // namespace

CorrespondencesPnP gather_input_PnPRansac(const std::vector<cv::Point3f>& candidate,
                                          const std::vector<cv::KeyPoint>& query,
                                          const std::vector<MatchDescription>& matches) {
  CorrespondencesPnP out{};

  for (const auto& match : matches) {
    if (!match.valid) continue;
    if (match.cand_index < 0 || match.cand_index >= static_cast<int>(candidate.size()))
      throw std::invalid_argument("cand_index out of range");
    if (match.query_index < 0 || match.query_index >= static_cast<int>(query.size()))
      throw std::invalid_argument("query_index out of range");

    if (!std::isfinite(candidate[match.cand_index].z) || candidate[match.cand_index].z <= 0)
      continue;
    out.object_points.push_back(candidate[match.cand_index]);
    out.image_points.push_back((query[match.query_index]).pt);
    out.candidate_ids.push_back(match.cand_index);
    out.query_ids.push_back(match.query_index);
  }

  return out;
}

PnpResult run_PnPRansac(const CorrespondencesPnP& data, const cv::Mat& K,
                        const cv::Mat& dist_coeffs, const PnpParams& params) {
  PnpResult result{};

  if (data.object_points.size() != data.image_points.size())
    throw std::invalid_argument("object_points and image_points must have the same size");

  if (data.image_points.size() < 4) return result;

  switch (params.backend) {
    case PnpBackend::Own:
      throw std::logic_error("Own backend not implemented");
      // pnp_own();
      // break;
    case PnpBackend::OpenCv:
      return pnp_opencv(data, K, dist_coeffs, params);
  }
  throw std::logic_error("unknown PnpBackend value");
}
}  // namespace pr