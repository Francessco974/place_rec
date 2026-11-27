// Tests for pnp.hpp: gather_input_PnPRansac (hand-made data) and
// run_PnPRansac (synthetic scenes with a known ground-truth pose).
#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
#include <limits>
#include <opencv2/calib3d.hpp>
#include <opencv2/core.hpp>
#include <set>
#include <stdexcept>
#include <vector>

#include "place_rec/pnp.hpp"

namespace {

using pr::CorrespondencesPnP;
using pr::MatchDescription;
using pr::PnpBackend;
using pr::PnpParams;
using pr::PnpResult;

constexpr float kNaN = std::numeric_limits<float>::quiet_NaN();
constexpr int kWidth = 640;
constexpr int kHeight = 480;

// TUM default intrinsics (fr1/fr2/fr3 "ROS default").
cv::Mat make_K() {
  return (cv::Mat_<double>(3, 3) << 525.0, 0.0, 319.5, 0.0, 525.0, 239.5, 0.0, 0.0, 1.0);
}

MatchDescription make_match(int query_index, int cand_index, bool valid) {
  MatchDescription m{};
  m.query_index = query_index;
  m.cand_index = cand_index;
  m.valid = valid;
  return m;
}

std::vector<cv::KeyPoint> make_query_keypoints(int n) {
  std::vector<cv::KeyPoint> kps;
  for (int i = 0; i < n; ++i) kps.emplace_back(cv::Point2f(10.0f * i, 20.0f * i), 31.0f);
  return kps;
}

// Angle (degrees) of R_est * R_gt^T, i.e. the rotation between the two estimates.
double rotation_error_deg(const cv::Mat& rvec_est, const cv::Mat& rvec_gt) {
  cv::Mat R_est, R_gt, r_err;
  cv::Rodrigues(rvec_est, R_est);
  cv::Rodrigues(rvec_gt, R_gt);
  cv::Rodrigues(R_est * R_gt.t(), r_err);
  return cv::norm(r_err) * 180.0 / CV_PI;
}

double translation_error_m(const cv::Mat& tvec_est, const cv::Mat& tvec_gt) {
  return cv::norm(tvec_est - tvec_gt);
}

// A synthetic scene: candidate-frame 3D points, query pixels, and which entries are true inliers.
struct Scene {
  CorrespondencesPnP data;
  std::vector<bool> is_inlier;
  cv::Mat rvec_gt;
  cv::Mat tvec_gt;
};

// n points spread like an indoor TUM view (x,y in [-1.5,1.5] m, depth 1-4 m), projected with a
// known pose (small loop-closure-like motion). Outliers get a uniformly random pixel, which is
// what a wrong descriptor match looks like. Deterministic through `seed`.
Scene make_scene(int n, double outlier_fraction, double noise_px, uint64_t seed) {
  cv::RNG rng(seed);
  Scene s;
  s.rvec_gt = (cv::Mat_<double>(3, 1) << 0.05, -0.20, 0.03);  // ~12 deg, mostly yaw
  s.tvec_gt = (cv::Mat_<double>(3, 1) << 0.30, -0.05, 0.20);  // ~0.36 m

  for (int i = 0; i < n; ++i) {
    s.data.object_points.emplace_back(static_cast<float>(rng.uniform(-1.5, 1.5)),
                                      static_cast<float>(rng.uniform(-1.5, 1.5)),
                                      static_cast<float>(rng.uniform(1.0, 4.0)));
  }
  std::vector<cv::Point2f> projected;
  cv::projectPoints(s.data.object_points, s.rvec_gt, s.tvec_gt, make_K(), cv::noArray(), projected);

  const int n_outliers = static_cast<int>(std::round(outlier_fraction * n));
  for (int i = 0; i < n; ++i) {
    const bool inlier = i >= n_outliers;  // first n_outliers entries are outliers
    cv::Point2f px = projected[i];
    if (inlier) {
      px.x += static_cast<float>(rng.gaussian(noise_px));
      px.y += static_cast<float>(rng.gaussian(noise_px));
    } else {
      px = {static_cast<float>(rng.uniform(0.0, double(kWidth))),
            static_cast<float>(rng.uniform(0.0, double(kHeight)))};
    }
    s.data.image_points.push_back(px);
    s.data.candidate_ids.push_back(i);
    s.data.query_ids.push_back(i);
    s.is_inlier.push_back(inlier);
  }
  return s;
}

// ---------------------------------------------------------------- gather

TEST(GatherPnp, KeepsOnlyValidMatchesWithFiniteDepth) {
  const std::vector<cv::Point3f> cand = {
      {0.1f, 0.2f, 1.0f},  // 0: good
      {0.0f, 0.0f, kNaN},  // 1: depth hole
      {0.3f, 0.1f, 0.0f},  // 2: z == 0
      {0.2f, 0.2f, -1.f},  // 3: z < 0
      {0.5f, 0.4f, 2.0f},  // 4: good, but its match is invalid
      {0.7f, 0.1f, 3.0f},  // 5: good
  };
  const auto query = make_query_keypoints(6);
  const std::vector<MatchDescription> matches = {
      make_match(0, 0, true), make_match(1, 1, true),  make_match(2, 2, true),
      make_match(3, 3, true), make_match(4, 4, false), make_match(5, 5, true),
  };

  const CorrespondencesPnP out = pr::gather_input_PnPRansac(cand, query, matches);

  ASSERT_EQ(out.object_points.size(), 2u);
  EXPECT_EQ(out.image_points.size(), 2u);
  EXPECT_EQ(out.candidate_ids, (std::vector<int>{0, 5}));
  EXPECT_EQ(out.query_ids, (std::vector<int>{0, 5}));
}

TEST(GatherPnp, PairsQueryPixelWithCandidatePointByIndex) {
  // Crossed indices: query row 0 matches candidate 2, query row 2 matches candidate 0.
  const std::vector<cv::Point3f> cand = {{1, 1, 1}, {2, 2, 2}, {3, 3, 3}};
  const auto query = make_query_keypoints(3);
  const std::vector<MatchDescription> matches = {make_match(0, 2, true), make_match(2, 0, true)};

  const CorrespondencesPnP out = pr::gather_input_PnPRansac(cand, query, matches);

  ASSERT_EQ(out.object_points.size(), 2u);
  EXPECT_EQ(out.object_points[0], cand[2]);
  EXPECT_EQ(out.image_points[0], query[0].pt);
  EXPECT_EQ(out.object_points[1], cand[0]);
  EXPECT_EQ(out.image_points[1], query[2].pt);
  EXPECT_EQ(out.candidate_ids, (std::vector<int>{2, 0}));
  EXPECT_EQ(out.query_ids, (std::vector<int>{0, 2}));
}

TEST(GatherPnp, EmptyMatchesGiveEmptyOutput) {
  const CorrespondencesPnP out = pr::gather_input_PnPRansac({}, {}, {});
  EXPECT_TRUE(out.object_points.empty());
  EXPECT_TRUE(out.image_points.empty());
  EXPECT_TRUE(out.candidate_ids.empty());
  EXPECT_TRUE(out.query_ids.empty());
}

TEST(GatherPnp, ThrowsOnOutOfRangeIndices) {
  const std::vector<cv::Point3f> cand = {{0, 0, 1}, {0, 0, 1}};
  const auto query = make_query_keypoints(2);
  EXPECT_THROW(pr::gather_input_PnPRansac(cand, query, {make_match(0, 2, true)}),
               std::invalid_argument);
  EXPECT_THROW(pr::gather_input_PnPRansac(cand, query, {make_match(0, -1, true)}),
               std::invalid_argument);
  EXPECT_THROW(pr::gather_input_PnPRansac(cand, query, {make_match(2, 0, true)}),
               std::invalid_argument);
  EXPECT_THROW(pr::gather_input_PnPRansac(cand, query, {make_match(-1, 0, true)}),
               std::invalid_argument);
}

TEST(GatherPnp, InvalidMatchIsSkippedBeforeIndexCheck) {
  // Documents current behaviour: indices of rejected matches are not checked.
  const std::vector<cv::Point3f> cand = {{0, 0, 1}};
  const auto query = make_query_keypoints(1);
  EXPECT_NO_THROW(pr::gather_input_PnPRansac(cand, query, {make_match(5, 99, false)}));
}

// ---------------------------------------------------------------- run_PnPRansac: contract

TEST(RunPnp, ThrowsOnSizeMismatch) {
  CorrespondencesPnP data;
  data.object_points.assign(5, {0, 0, 1});
  data.image_points.assign(4, {0, 0});
  EXPECT_THROW(pr::run_PnPRansac(data, make_K(), cv::Mat(), PnpParams{}), std::invalid_argument);
}

TEST(RunPnp, FewerThanFourPointsIsAFailedResultNotAnError) {
  Scene s = make_scene(3, 0.0, 0.0, 1);
  PnpResult r;
  ASSERT_NO_THROW(r = pr::run_PnPRansac(s.data, make_K(), cv::Mat(), PnpParams{}));
  EXPECT_FALSE(r.success);
  EXPECT_TRUE(r.inliers.empty());
  EXPECT_TRUE(r.rvec.empty());
  EXPECT_TRUE(r.tvec.empty());
}

TEST(RunPnp, OwnBackendThrowsUntilImplemented) {
  Scene s = make_scene(50, 0.0, 0.0, 1);
  PnpParams p;
  p.backend = PnpBackend::Own;
  EXPECT_THROW(pr::run_PnPRansac(s.data, make_K(), cv::Mat(), p), std::logic_error);
}

// ---------------------------------------------------------------- run_PnPRansac: accuracy

TEST(RunPnp, NoiseFreeRecoversExactPoseAndAllInliers) {
  Scene s = make_scene(100, 0.0, 0.0, 42);
  PnpParams p;
  p.reprojection_error_px = 1.0f;

  const PnpResult r = pr::run_PnPRansac(s.data, make_K(), cv::Mat(), p);

  ASSERT_TRUE(r.success);
  EXPECT_LT(rotation_error_deg(r.rvec, s.rvec_gt), 0.01);
  EXPECT_LT(translation_error_m(r.tvec, s.tvec_gt), 1e-3);
  EXPECT_EQ(r.inliers.size(), 100u);
}

TEST(RunPnp, NoisyWithOutliersRecoversPoseAndSeparatesInliers) {
  // 1 px noise (ORB at level 0), 30 % random outliers, tau = 4 px.
  Scene s = make_scene(200, 0.30, 1.0, 7);
  PnpParams p;
  p.reprojection_error_px = 4.0f;

  const PnpResult r = pr::run_PnPRansac(s.data, make_K(), cv::Mat(), p);

  ASSERT_TRUE(r.success);
  EXPECT_LT(rotation_error_deg(r.rvec, s.rvec_gt), 0.5);
  EXPECT_LT(translation_error_m(r.tvec, s.tvec_gt), 0.03);

  const int n_true = static_cast<int>(std::count(s.is_inlier.begin(), s.is_inlier.end(), true));

  // The returned inliers are recounted with the REFINED pose (OpenCV's own list comes from the
  // best 4-point hypothesis and missed 12 of 140 here). 2D Gaussian, sigma 1 px:
  // P(|e| > 4 px) = exp(-8) ~ 3e-4, so almost all true inliers must be back.
  int true_pos = 0, false_pos = 0;
  for (int idx : r.inliers) (s.is_inlier[idx] ? true_pos : false_pos)++;
  EXPECT_GE(true_pos, static_cast<int>(0.97 * n_true));
  // A random pixel lands within 4 px of the right spot with p ~ pi*16/(640*480) ~ 2e-4.
  EXPECT_LE(false_pos, 1);

  // The returned list must equal an independent recount with the returned pose.
  std::vector<cv::Point2f> proj;
  cv::projectPoints(s.data.object_points, r.rvec, r.tvec, make_K(), cv::noArray(), proj);
  std::vector<int> recount;
  for (size_t i = 0; i < proj.size(); ++i)
    if (cv::norm(proj[i] - s.data.image_points[i]) < p.reprojection_error_px)
      recount.push_back(static_cast<int>(i));
  EXPECT_EQ(r.inliers, recount);
}

TEST(RunPnp, PointsBehindTheCameraAreNeverInliers) {
  // A pixel cannot tell X from -X: both project to the same place. Here point 0 is mirrored
  // through the query camera centre, so two poses explain all 50 pixels equally well:
  //   - the true pose (point 0 behind the camera), and
  //   - a ~180 deg "mirror" pose (the other 49 behind the camera).
  // OpenCV scores only reprojection, so it may return either (measured: the mirror one).
  // The cheirality check in the recount must reject every point behind the camera, so the
  // count is 49 or 0, never 50.
  Scene s = make_scene(50, 0.0, 0.0, 5);
  cv::Mat R;
  cv::Rodrigues(s.rvec_gt, R);
  const cv::Point3f& X0 = s.data.object_points[0];
  cv::Mat Xq = R * (cv::Mat_<double>(3, 1) << X0.x, X0.y, X0.z) + s.tvec_gt;  // query frame
  cv::Mat Xc = R.t() * (-Xq - s.tvec_gt);  // mirrored point, back in the candidate frame
  s.data.object_points[0] =
      cv::Point3f(static_cast<float>(Xc.at<double>(0)), static_cast<float>(Xc.at<double>(1)),
                  static_cast<float>(Xc.at<double>(2)));
  PnpParams p;
  p.reprojection_error_px = 2.0f;

  const PnpResult r = pr::run_PnPRansac(s.data, make_K(), cv::Mat(), p);

  ASSERT_TRUE(r.success);
  EXPECT_TRUE(r.inliers.size() == 49u || r.inliers.empty()) << r.inliers.size();
  // Independent check: every returned inlier is in front of the camera with the returned pose.
  cv::Mat R_est;
  cv::Rodrigues(r.rvec, R_est);
  for (int idx : r.inliers) {
    const cv::Point3f& X = s.data.object_points[idx];
    cv::Mat Xq_est = R_est * (cv::Mat_<double>(3, 1) << X.x, X.y, X.z) + r.tvec;
    EXPECT_GT(Xq_est.at<double>(2), 0.0) << "point " << idx << " behind the camera";
  }
}

TEST(RunPnp, IsDeterministicAcrossCalls) {
  Scene s = make_scene(200, 0.40, 1.0, 3);
  PnpParams p;
  p.reprojection_error_px = 4.0f;
  const PnpResult a = pr::run_PnPRansac(s.data, make_K(), cv::Mat(), p);
  const PnpResult b = pr::run_PnPRansac(s.data, make_K(), cv::Mat(), p);
  ASSERT_EQ(a.success, b.success);
  EXPECT_EQ(a.inliers, b.inliers);
  if (a.success) {
    EXPECT_EQ(cv::norm(a.rvec - b.rvec), 0.0);
  }
}

// ---------------------------------------------------------------- negative: no geometric link

TEST(RunPnp, PureOutliersGiveOnlyALuckFloorOfInliers) {
  // Every pixel is random: this is what a false loop looks like to the verifier.
  // The inlier count must stay tiny compared with the true-loop case above (~140).
  Scene s = make_scene(200, 1.0, 0.0, 11);
  PnpParams p;  // OpenCV defaults: tau = 8 px, 100 iterations

  const PnpResult r = pr::run_PnPRansac(s.data, make_K(), cv::Mat(), p);

  const size_t n_inliers = r.success ? r.inliers.size() : 0u;
  EXPECT_LE(n_inliers, 12u) << "luck floor too high: " << n_inliers;
  RecordProperty("pure_outlier_inliers", static_cast<int>(n_inliers));
}

}  // namespace