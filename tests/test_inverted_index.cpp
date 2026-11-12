#include <gtest/gtest.h>

#include <cmath>
#include <random>
#include <stdexcept>
#include <vector>

#include "place_rec/bow_vector.hpp"
#include "place_rec/inverted_index.hpp"
#include "place_rec/vocabulary_tree.hpp"

using BV = pr::VocabularyTree::BowVector;

TEST(InvertedIndex, ToyExample) {
  pr::InvertedIndex idx(8);
  EXPECT_EQ(idx.add({{2, 0.5f}, {7, 0.5f}}), 0);
  EXPECT_EQ(idx.add({{5, 1.0f}}), 1);
  EXPECT_EQ(idx.add({{2, 0.25f}, {5, 0.5f}, {7, 0.25f}}), 2);
  const auto s = idx.scores({{2, 0.5f}, {7, 0.5f}});
  ASSERT_EQ(s.size(), 3u);
  EXPECT_NEAR(s[0], 1.0f, 1e-6f);
  EXPECT_NEAR(s[1], 0.0f, 1e-6f);
  EXPECT_NEAR(s[2], 0.5f, 1e-6f);
}

TEST(InvertedIndex, EmptyQueryGivesZeros) {
  pr::InvertedIndex idx(4);
  idx.add({{1, 1.0f}});
  const auto s = idx.scores({});
  ASSERT_EQ(s.size(), 1u);
  EXPECT_FLOAT_EQ(s[0], 0.0f);
}

TEST(InvertedIndex, BadWordThrows) {
  pr::InvertedIndex idx(4);
  EXPECT_THROW(idx.add({{9, 1.0f}}), std::out_of_range);
}

TEST(InvertedIndex, MatchesSimilarityScore) {
  std::mt19937 gen(3);
  const int W = 30;
  auto random_bow = [&] {
    std::vector<float> w(W, 0.0f);
    std::uniform_int_distribution<int> n(1, 8), word(0, W - 1);
    std::uniform_real_distribution<float> val(0.1f, 1.0f);
    for (int i = n(gen); i > 0; --i) w[word(gen)] += val(gen);
    float sum = 0;
    for (float x : w) sum += x;
    BV v;
    for (int i = 0; i < W; ++i)
      if (w[i] > 0) v.emplace_back(i, w[i] / sum);
    return v;
  };
  pr::InvertedIndex idx(W);
  std::vector<BV> kfs;
  for (int i = 0; i < 50; ++i) {
    kfs.push_back(random_bow());
    idx.add(kfs.back());
  }
  for (int q = 0; q < 20; ++q) {
    const BV query = random_bow();
    const auto s = idx.scores(query);
    for (size_t i = 0; i < kfs.size(); ++i)
      EXPECT_NEAR(s[i], pr::similarity_score(query, kfs[i]), 1e-5f);
  }
}