#pragma once
#include <utility>
#include <vector>

namespace pr {

// Sparse BoW vector, sorted by word id.
// .first is the word id, .second is the weight of that word in this image.
using BowVector = std::vector<std::pair<int, float>>;

// Histogram intersection of two BoW vectors.
// Precondition: both sorted by word id, L1-normalized, weights >= 0.
// Returns a value in [0, 1]; 0 if either vector is empty. Symmetric.
float similarity_score(const BowVector& bow1, const BowVector& bow2);

}  // namespace pr