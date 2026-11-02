#pragma once
#include <random>
#include <vector>

#include "place_rec/binary_descriptor.hpp"

namespace pr {

class VocabularyTree {
 public:
  struct KmeansOutput {
    std::vector<Desc256> centroids;  // size <= k (empty clusters removed)
    std::vector<int> assign;         // assign[i] = index into centroids
  };

  static KmeansOutput kmeans(const std::vector<const Desc256*>& x, int k, int max_iters,
                             std::mt19937& rng);
};

}  // namespace pr