#pragma once
#include <random>
#include <utility>
#include <vector>

#include "place_rec/binary_descriptor.hpp"

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

  // .first is the word id, .second is the weight of that word in this image
  using BowVector = std::vector<std::pair<int, float>>;

  static KmeansOutput kmeans(const std::vector<const Desc256*>& x, int k, int max_iters,
                             std::mt19937& rng);

  void train(const std::vector<const Desc256*>& x, const Params& params,
             const std::vector<int>& imgs_ids);
  int transform(const Desc256& descriptor) const;

  int num_words() const { return static_cast<int>(word_node_.size()); }
  int num_nodes() const { return static_cast<int>(tree_.size()); }
  const Node& node(int i) const { return tree_.at(i); }
  int word_node(int w) const { return word_node_.at(w); }
  float idf(int w) const { return idf_.at(w); }

  BowVector bow(const std::vector<const Desc256*>& descs) const;

 private:
  std::vector<Node> tree_;
  Params params_;
  std::vector<int> word_node_;
  std::vector<float> idf_;

  void build(int node, const std::vector<const Desc256*>& x, int depth, std::mt19937& rng);
};

// Histogram intersection of two BoW vectors.
// Precondition: both sorted by word id, L1-normalized, weights >= 0.
// Returns a value in [0, 1]; 0 if either vector is empty. Symmetric.
float similarity_score(const VocabularyTree::BowVector& bow1,
                       const VocabularyTree::BowVector& bow2);

class InvertedIndex {
 public:
  explicit InvertedIndex(int num_words) : lists_(num_words) {}

  int add(const VocabularyTree::BowVector& v);
  std::vector<float> scores(const VocabularyTree::BowVector& v) const;

 private:
  // word [keyframe [keyframe id - weigth]]
  std::vector<std::vector<std::pair<int, float>>> lists_;

  int num_keyframes_ = 0;
};

}  // namespace pr