#include "place_rec/inverted_index.hpp"

#include <algorithm>

namespace pr {

int InvertedIndex::add(const BowVector& v) {
  for (const auto& [word, weight] : v) {
    lists_.at(word).emplace_back(num_keyframes_, weight);
  }
  return num_keyframes_++;
}

std::vector<float> InvertedIndex::scores(const BowVector& q) const {
  std::vector<float> result(num_keyframes_, 0.0f);
  for (const auto& [word, weight_query] : q) {
    for (const auto& [key_id, weight_bow] : lists_.at(word)) {
      result[key_id] += std::min(weight_bow, weight_query);
    }
  }
  return result;
}

}  // namespace pr