#include <gtest/gtest.h>

#include <cmath>
#include <random>
#include <set>
#include <stdexcept>
#include <vector>

#include "place_rec/vocabulary_tree.hpp"

namespace {
using pr::Desc256;
using pr::VocabularyTree;

// base with `flips` random bits flipped (a bit may be hit twice: fine)
Desc256 noisy(const Desc256& base, int flips, std::mt19937& gen) {
  Desc256 d = base;
  std::uniform_int_distribution<int> bit(0, 255);
  for (int f = 0; f < flips; ++f) {
    const int b = bit(gen);
    d[b / 64] ^= (1ULL << (b % 64));
  }
  return d;
}

// Call only AFTER `v` is fully built, or push_back may invalidate the pointers.
std::vector<const Desc256*> ptrs(const std::vector<Desc256>& v) {
  std::vector<const Desc256*> p;
  p.reserve(v.size());
  for (const auto& d : v) p.push_back(&d);
  return p;
}

pr::Desc256 zeros() { return pr::Desc256{}; }
pr::Desc256 ones() {
  pr::Desc256 d;
  d.fill(~0ULL);
  return d;
}
pr::Desc256 with_low8(pr::Desc256 d, bool set) {
  if (set)
    d[0] |= 0xFFULL;
  else
    d[0] &= ~0xFFULL;
  return d;
}

void expect_leaf_invariant(const pr::VocabularyTree& t) {
  for (int w = 0; w < t.num_words(); ++w) {
    const auto& n = t.node(t.word_node(w));
    EXPECT_EQ(n.word, w);
    EXPECT_TRUE(n.children.empty());
  }
}
}  // namespace

TEST(KMeans, TwoSeparatedClusters) {
  std::mt19937 gen(123);
  const Desc256 zeros_d{};
  Desc256 ones_d;
  ones_d.fill(~0ULL);
  std::vector<Desc256> data;
  for (int i = 0; i < 10; ++i) data.push_back(noisy(zeros_d, 1 + i % 3, gen));
  for (int i = 0; i < 10; ++i) data.push_back(noisy(ones_d, 1 + i % 3, gen));
  const auto x = ptrs(data);

  for (unsigned seed = 0; seed < 20; ++seed) {
    std::mt19937 rng(seed);
    const auto r = VocabularyTree::kmeans(x, 2, 20, rng);
    ASSERT_EQ(r.centroids.size(), 2u) << "seed " << seed;
    ASSERT_EQ(r.assign.size(), 20u);
    for (int i = 1; i < 10; ++i) EXPECT_EQ(r.assign[i], r.assign[0]) << "seed " << seed;
    for (int i = 11; i < 20; ++i) EXPECT_EQ(r.assign[i], r.assign[10]) << "seed " << seed;
    EXPECT_NE(r.assign[0], r.assign[10]) << "seed " << seed;
  }
}

TEST(KMeans, FewerPointsThanK) {
  std::mt19937 gen(1), rng(1);
  std::vector<Desc256> data;
  for (int i = 0; i < 3; ++i) data.push_back(noisy(Desc256{}, 20 * i, gen));
  const auto r = VocabularyTree::kmeans(ptrs(data), 5, 10, rng);
  EXPECT_EQ(r.centroids.size(), 3u);
  EXPECT_EQ(r.assign, (std::vector<int>{0, 1, 2}));
}

TEST(KMeans, IdenticalPointsGiveOneCluster) {
  std::mt19937 rng(7);
  std::vector<Desc256> data(5, Desc256{});
  const auto r = VocabularyTree::kmeans(ptrs(data), 3, 10, rng);
  EXPECT_EQ(r.centroids.size(), 1u);
  for (int a : r.assign) EXPECT_EQ(a, 0);
}

TEST(KMeans, InvalidKThrows) {
  std::mt19937 rng(0);
  std::vector<Desc256> data(4, Desc256{});
  EXPECT_THROW(VocabularyTree::kmeans(ptrs(data), 0, 10, rng), std::invalid_argument);
}

TEST(KMeans, EmptyInput) {
  std::mt19937 rng(0);
  const auto r = VocabularyTree::kmeans({}, 3, 10, rng);
  EXPECT_TRUE(r.centroids.empty());
  EXPECT_TRUE(r.assign.empty());
}

TEST(KMeans, LabelsAreValidAndDeterministic) {
  std::mt19937 gen(5);
  std::vector<Desc256> data;
  for (int i = 0; i < 200; ++i) data.push_back(noisy(Desc256{}, 100, gen));
  const auto x = ptrs(data);
  std::mt19937 r1(42), r2(42);
  const auto a = VocabularyTree::kmeans(x, 5, 20, r1);
  const auto b = VocabularyTree::kmeans(x, 5, 20, r2);
  for (int l : a.assign) {
    EXPECT_GE(l, 0);
    EXPECT_LT(l, static_cast<int>(a.centroids.size()));
  }
  EXPECT_EQ(a.assign, b.assign);
  EXPECT_EQ(a.centroids, b.centroids);
}

TEST(VocabularyTreeTrain, TwoLevelsOfClusters) {
  // A1 = zeros, A2 = zeros + 8 bits, B1 = ones, B2 = ones - 8 bits; two copies each
  const auto a1 = zeros(), a2 = with_low8(zeros(), true);
  const auto b1 = ones(), b2 = with_low8(ones(), false);
  std::vector<pr::Desc256> data = {a1, a1, a2, a2, b1, b1, b2, b2};
  pr::VocabularyTree t;
  t.train(ptrs(data), {2, 2, 25, 42}, std::vector<int>(data.size(), 0));
  EXPECT_EQ(t.num_nodes(), 7);
  EXPECT_EQ(t.num_words(), 4);
  expect_leaf_invariant(t);
}

TEST(VocabularyTreeTrain, IdenticalPointsGiveRootLeaf) {
  std::vector<pr::Desc256> data(5, ones());
  pr::VocabularyTree t;
  t.train(ptrs(data), {2, 2, 25, 42}, std::vector<int>(data.size(), 0));
  EXPECT_EQ(t.num_nodes(), 1);
  EXPECT_EQ(t.num_words(), 1);
  expect_leaf_invariant(t);
}

TEST(VocabularyTreeTrain, EmptyInputGivesNoWords) {
  pr::VocabularyTree t;
  t.train({}, {}, {});
  EXPECT_EQ(t.num_nodes(), 1);
  EXPECT_EQ(t.num_words(), 0);
}

TEST(VocabularyTreeTransform, TrainingPointsMapToDistinctLeaves) {
  const auto a1 = zeros(), a2 = with_low8(zeros(), true);
  const auto b1 = ones(), b2 = with_low8(ones(), false);
  std::vector<pr::Desc256> data = {a1, a1, a2, a2, b1, b1, b2, b2};
  pr::VocabularyTree t;
  t.train(ptrs(data), {2, 2, 25, 42}, std::vector<int>(data.size(), 0));
  for (const auto& d : data) {
    const int w = t.transform(d);
    ASSERT_GE(w, 0);
    ASSERT_LT(w, t.num_words());
  }
  EXPECT_NE(t.transform(a1), t.transform(a2));
  EXPECT_NE(t.transform(a1), t.transform(b1));
  EXPECT_NE(t.transform(b1), t.transform(b2));
  EXPECT_EQ(t.transform(a1), t.transform(a1));
}

TEST(VocabularyTreeTransform, RootOnlyTreeReturnsWordZero) {
  std::vector<pr::Desc256> data(5, ones());
  pr::VocabularyTree t;
  t.train(ptrs(data), {2, 2, 25, 42}, std::vector<int>(data.size(), 0));
  EXPECT_EQ(t.transform(zeros()), 0);
  EXPECT_EQ(t.transform(ones()), 0);
}

TEST(VocabularyTreeTransform, EmptyTreeReturnsMinusOne) {
  pr::VocabularyTree untrained;
  EXPECT_EQ(untrained.transform(zeros()), -1);  // tree_ empty
  pr::VocabularyTree t;
  t.train({}, {}, {});
  EXPECT_EQ(t.transform(zeros()), -1);  // root without word
}

TEST(VocabularyTreeTransform, NoisyQueryKeepsWord) {
  std::mt19937 gen(9);
  const auto a1 = zeros(), a2 = with_low8(zeros(), true);
  const auto b1 = ones(), b2 = with_low8(ones(), false);
  std::vector<pr::Desc256> data = {a1, a1, a2, a2, b1, b1, b2, b2};
  pr::VocabularyTree t;
  t.train(ptrs(data), {2, 2, 25, 42}, std::vector<int>(data.size(), 0));
  for (const auto& base : data) {
    for (int i = 0; i < 20; ++i) EXPECT_EQ(t.transform(noisy(base, 3, gen)), t.transform(base));
  }
}

TEST(VocabularyTreeIdf, MatchesSetBasedCount) {
  const auto a1 = zeros(), a2 = with_low8(zeros(), true);
  const auto b1 = ones(), b2 = with_low8(ones(), false);
  std::vector<pr::Desc256> data = {a1, a1, a1, a2, a1, b1, a1, b2};
  const std::vector<int> ids = {0, 0, 0, 0, 1, 1, 2, 2};
  pr::VocabularyTree t;
  t.train(ptrs(data), {2, 2, 25, 42}, ids);
  ASSERT_GE(t.num_words(), 2);

  std::vector<std::set<int>> seen(3);
  for (size_t i = 0; i < data.size(); ++i) seen[ids[i]].insert(t.transform(data[i]));
  for (int w = 0; w < t.num_words(); ++w) {
    int df = 0;
    for (const auto& s : seen) df += static_cast<int>(s.count(w));
    ASSERT_GT(df, 0);
    EXPECT_FLOAT_EQ(t.idf(w), std::log(3.0f / df));
  }
}

TEST(VocabularyTreeIdf, SizeMismatchThrows) {
  std::vector<pr::Desc256> data(4, ones());
  pr::VocabularyTree t;
  EXPECT_THROW(t.train(ptrs(data), {}, {0, 0}), std::invalid_argument);
}