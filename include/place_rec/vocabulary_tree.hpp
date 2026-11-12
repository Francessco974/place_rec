#pragma once
#include <random>
#include <string>
#include <vector>

#include "place_rec/binary_descriptor.hpp"
#include "place_rec/bow_vector.hpp"

namespace pr {

class VocabularyTree {
 public:
  struct KmeansOutput {
    std::vector<Desc256> centroids;  // size <= k (empty clusters removed)
    std::vector<int> assign;         // assign[i] = index into centroids
  };

  struct Node {
    Desc256 centroid;
    int word{-1};
    std::vector<int> children;
  };
  struct Params {
    int k = 10;
    int L = 6;
    int max_iters = 25;
    unsigned seed = 42;
  };

  // kept so existing code using VocabularyTree::BowVector still compiles
  using BowVector = ::pr::BowVector;

  static KmeansOutput kmeans(const std::vector<const Desc256*>& x, int k, int max_iters,
                             std::mt19937& rng);

  // imgs_ids[i] is the image of x[i]; ids >= 0, non-decreasing (contiguous per image).
  void train(const std::vector<const Desc256*>& x, const Params& params,
             const std::vector<int>& imgs_ids);

  // Word id of a descriptor, or -1 if the vocabulary has no words.
  int transform(const Desc256& descriptor) const;

  int num_words() const { return static_cast<int>(word_node_.size()); }
  int num_nodes() const { return static_cast<int>(tree_.size()); }
  const Node& node(int i) const { return tree_.at(i); }
  int word_node(int w) const { return word_node_.at(w); }
  float idf(int w) const { return idf_.at(w); }

  // Sorted by word id, sums to 1, empty if no informative words.
  BowVector bow(const std::vector<const Desc256*>& descs) const;

  // Binary file, little-endian, format version 1. Throws std::runtime_error on any problem.
  void save(const std::string& path) const;
  static VocabularyTree load(const std::string& path);

 private:
  std::vector<Node> tree_;
  Params params_;
  std::vector<int> word_node_;
  std::vector<float> idf_;

  void build(int node, const std::vector<const Desc256*>& x, int depth, std::mt19937& rng);
};

}  // namespace pr