#include <gtest/gtest.h>

#include <random>
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
}  // namespace

TEST(KMeans, TwoSeparatedClusters) {
  std::mt19937 gen(123);
  const Desc256 zeros{};
  Desc256 ones;
  ones.fill(~0ULL);
  std::vector<Desc256> data;
  for (int i = 0; i < 10; ++i) data.push_back(noisy(zeros, 1 + i % 3, gen));
  for (int i = 0; i < 10; ++i) data.push_back(noisy(ones, 1 + i % 3, gen));
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