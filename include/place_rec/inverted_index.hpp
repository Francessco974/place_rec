#pragma once
#include <utility>
#include <vector>

#include "place_rec/bow_vector.hpp"

namespace pr {

class InvertedIndex {
 public:
  explicit InvertedIndex(int num_words) : lists_(num_words) {}

  // Precondition: v sorted by word id, every word < num_words (else std::out_of_range).
  // Returns the new keyframe id. Every call consumes an id, even for an empty vector.
  int add(const BowVector& v);

  // One score per keyframe id (histogram intersection with q); 0 if no shared word.
  // Same preconditions as add.
  std::vector<float> scores(const BowVector& q) const;

 private:
  // lists_[word] = list of (keyframe id, weight of that word in that keyframe)
  std::vector<std::vector<std::pair<int, float>>> lists_;
  int num_keyframes_ = 0;
};

}  // namespace pr