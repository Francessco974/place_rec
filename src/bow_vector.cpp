#include "place_rec/bow_vector.hpp"

#include <algorithm>
#include <cstddef>

namespace pr {

float similarity_score(const BowVector& bow1, const BowVector& bow2) {
  float score = 0.0f;
  std::size_t i = 0, j = 0;
  while (i < bow1.size() && j < bow2.size()) {
    if (bow1[i].first > bow2[j].first)
      j++;
    else if (bow2[j].first > bow1[i].first)
      i++;
    else {
      score += std::min(bow1[i].second, bow2[j].second);
      ++i;
      ++j;
    }
  }
  return score;
}

}  // namespace pr