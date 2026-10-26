#pragma once
#include <vector>

#include "place_rec/types.hpp"

namespace pr {

class PlaceRecognizer {
 public:
  virtual ~PlaceRecognizer() = default;

  virtual std::vector<Candidate> query(const Keyframe& Kf, double max_stamp,
                                       int max_results) const = 0;

  virtual void add(const Keyframe& kf) = 0;
};

}  // namespace pr