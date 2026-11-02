#include "place_rec/vocabulary_tree.hpp"

#include <algorithm>
#include <cstddef>
#include <stdexcept>
#include <vector>

#include "place_rec/binary_descriptor.hpp"

namespace pr {

auto VocabularyTree::kmeans(const std::vector<const Desc256*>& x, int k, int max_iters,
                            std::mt19937& rng) -> KmeansOutput {
  KmeansOutput results;
  if (k <= 0) throw std::invalid_argument("k must be greater than 0");

  int point_num = static_cast<int>(x.size());
  if (!point_num) return results;

  if (point_num <= k) {
    results.centroids.reserve(point_num);
    results.assign.reserve(point_num);

    for (int i = 0; i < point_num; ++i) {
      results.centroids.push_back(*x[i]);
      results.assign.push_back(i);
    }

    return results;
  }

  // k-means++ seeding
  std::vector<int> dist(point_num);
  std::vector<double> weights(point_num);

  int idx = std::uniform_int_distribution<int>(0, point_num - 1)(rng);
  results.centroids.push_back(*x[idx]);  // store the first cluster

  for (int i = 0; i < point_num; i++) {
    dist[i] = hamming(*x[i], *x[idx]);
  }

  while (results.centroids.size() < static_cast<std::size_t>(k)) {
    double sum = 0;
    for (int i = 0; i < point_num; i++) {
      weights[i] = static_cast<double>(dist[i]) * static_cast<double>(dist[i]);
      sum += weights[i];
    }
    if (sum <= 1e-6) break;

    idx = std::discrete_distribution<int>(weights.begin(), weights.end())(rng);
    results.centroids.push_back(*x[idx]);

    for (int i = 0; i < point_num; i++) {
      dist[i] = std::min(dist[i], hamming(*x[i], results.centroids.back()));
    }
  }

  // Lloyd loop
  results.assign.assign(point_num, -1);
  for (int iter = 0; iter < max_iters; iter++) {
    const int c = static_cast<int>(results.centroids.size());  // may be < k
    bool changed = false;

    // assign: nearest centroid, ties go to the lower index
    for (int i = 0; i < point_num; i++) {
      int best = -1, best_dist = 257;  // 257 > any Hamming distance
      for (int j = 0; j < c; j++) {
        const int d = hamming(*x[i], results.centroids[j]);
        if (d < best_dist) {
          best_dist = d;
          best = j;
        }
      }
      if (best != results.assign[i]) {
        results.assign[i] = best;
        changed = true;
      }
    }
    if (!changed) break;

    // update: group in one pass, then majority vote
    std::vector<std::vector<const Desc256*>> members(c);
    for (int i = 0; i < point_num; i++) members[results.assign[i]].push_back(x[i]);
    for (int j = 0; j < c; j++)
      if (!members[j].empty()) results.centroids[j] = majority(members[j]);

    // compact: remap table, in-place copy, then rewrite assign
    std::vector<int> remap(c, -1);
    int next = 0;
    for (int j = 0; j < c; j++) {
      if (members[j].empty()) continue;
      remap[j] = next;
      results.centroids[next++] = results.centroids[j];
    }
    results.centroids.resize(next);
    for (int& a : results.assign) a = remap[a];
  }
  return results;
}
}  // namespace pr
