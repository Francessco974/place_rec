#pragma once
#include <opencv2/core/mat.hpp>
#include <vector>

#include "place_rec/place_recognizer.hpp"
#include "place_rec/types.hpp"

namespace pr {

class TinyImageRecognizer : public PlaceRecognizer {
 public:
  std::vector<Candidate> query(const Keyframe& kf, double max_stamp,
                               int max_results) const override;
  void add(const Keyframe& kf) override;

 private:
  static cv::Mat describe(const cv::Mat& gray);
  std::vector<cv::Mat> descs_;
  std::vector<double> stamps_;
};

}  // namespace pr