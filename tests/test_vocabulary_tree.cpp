#include <gtest/gtest.h>

#include <cmath>
#include <filesystem>
#include <fstream>
#include <iterator>
#include <random>
#include <set>
#include <stdexcept>
#include <vector>

#include "place_rec/vocabulary_tree.hpp"
#include "test_helpers.hpp"

using pr::Desc256;
using pr::VocabularyTree;
using namespace pr::test;

namespace {
void expect_leaf_invariant(const VocabularyTree& t) {
  for (int w = 0; w < t.num_words(); ++w) {
    const auto& n = t.node(t.word_node(w));
    EXPECT_EQ(n.word, w);
    EXPECT_TRUE(n.children.empty());
  }
}
}  // namespace

// ---------- k-means ----------

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

// ---------- train ----------

TEST(VocabularyTreeTrain, TwoLevelsOfClusters) {
  // A1 = zeros, A2 = zeros + 8 bits, B1 = ones, B2 = ones - 8 bits; two copies each
  const auto a1 = zeros(), a2 = with_low8(zeros(), true);
  const auto b1 = ones(), b2 = with_low8(ones(), false);
  std::vector<Desc256> data = {a1, a1, a2, a2, b1, b1, b2, b2};
  VocabularyTree t;
  t.train(ptrs(data), {2, 2, 25, 42}, std::vector<int>(data.size(), 0));
  EXPECT_EQ(t.num_nodes(), 7);
  EXPECT_EQ(t.num_words(), 4);
  expect_leaf_invariant(t);
}

TEST(VocabularyTreeTrain, IdenticalPointsGiveRootLeaf) {
  std::vector<Desc256> data(5, ones());
  VocabularyTree t;
  t.train(ptrs(data), {2, 2, 25, 42}, std::vector<int>(data.size(), 0));
  EXPECT_EQ(t.num_nodes(), 1);
  EXPECT_EQ(t.num_words(), 1);
  expect_leaf_invariant(t);
}

TEST(VocabularyTreeTrain, EmptyInputGivesNoWords) {
  VocabularyTree t;
  t.train({}, {}, {});
  EXPECT_EQ(t.num_nodes(), 1);
  EXPECT_EQ(t.num_words(), 0);
}

// ---------- transform ----------

TEST(VocabularyTreeTransform, TrainingPointsMapToDistinctLeaves) {
  const auto a1 = zeros(), a2 = with_low8(zeros(), true);
  const auto b1 = ones(), b2 = with_low8(ones(), false);
  std::vector<Desc256> data = {a1, a1, a2, a2, b1, b1, b2, b2};
  VocabularyTree t;
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
  std::vector<Desc256> data(5, ones());
  VocabularyTree t;
  t.train(ptrs(data), {2, 2, 25, 42}, std::vector<int>(data.size(), 0));
  EXPECT_EQ(t.transform(zeros()), 0);
  EXPECT_EQ(t.transform(ones()), 0);
}

TEST(VocabularyTreeTransform, EmptyTreeReturnsMinusOne) {
  VocabularyTree untrained;
  EXPECT_EQ(untrained.transform(zeros()), -1);  // tree_ empty
  VocabularyTree t;
  t.train({}, {}, {});
  EXPECT_EQ(t.transform(zeros()), -1);  // root without word
}

TEST(VocabularyTreeTransform, NoisyQueryKeepsWord) {
  std::mt19937 gen(9);
  const auto a1 = zeros(), a2 = with_low8(zeros(), true);
  const auto b1 = ones(), b2 = with_low8(ones(), false);
  std::vector<Desc256> data = {a1, a1, a2, a2, b1, b1, b2, b2};
  VocabularyTree t;
  t.train(ptrs(data), {2, 2, 25, 42}, std::vector<int>(data.size(), 0));
  for (const auto& base : data) {
    for (int i = 0; i < 20; ++i) EXPECT_EQ(t.transform(noisy(base, 3, gen)), t.transform(base));
  }
}

// ---------- IDF ----------

TEST(VocabularyTreeIdf, MatchesSetBasedCount) {
  const auto a1 = zeros(), a2 = with_low8(zeros(), true);
  const auto b1 = ones(), b2 = with_low8(ones(), false);
  std::vector<Desc256> data = {a1, a1, a1, a2, a1, b1, a1, b2};
  const std::vector<int> ids = {0, 0, 0, 0, 1, 1, 2, 2};
  VocabularyTree t;
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
  std::vector<Desc256> data(4, ones());
  VocabularyTree t;
  EXPECT_THROW(t.train(ptrs(data), {}, {0, 0}), std::invalid_argument);
}

TEST(VocabularyTreeIdf, UnsortedIdsThrow) {
  std::vector<Desc256> data(4, ones());
  VocabularyTree t;
  EXPECT_THROW(t.train(ptrs(data), {}, {0, 1, 0, 1}), std::invalid_argument);
}

TEST(VocabularyTreeIdf, NegativeIdThrows) {
  std::vector<Desc256> data(2, ones());
  VocabularyTree t;
  EXPECT_THROW(t.train(ptrs(data), {}, {-1, 0}), std::invalid_argument);
}

TEST(VocabularyTreeIdf, GapInIdsCountsGroups) {
  const auto a1 = zeros(), b1 = ones();
  std::vector<Desc256> data = {a1, b1, a1, b1};
  VocabularyTree t;
  t.train(ptrs(data), {2, 2, 25, 42}, {0, 0, 5, 5});  // 2 images, ids 0 and 5
  for (int w = 0; w < t.num_words(); ++w)
    EXPECT_FLOAT_EQ(t.idf(w), 0.0f);  // each word in both images
}

namespace {
std::string tmp_path(const char* name) {
  return (std::filesystem::temp_directory_path() / name).string();
}
VocabularyTree small_tree() {
  const auto a1 = zeros(), a2 = with_low8(zeros(), true);
  const auto b1 = ones(), b2 = with_low8(ones(), false);
  static const std::vector<Desc256> data = {a1, a1, a2, a2, a1, b1, b2, b2};
  VocabularyTree t;
  t.train(ptrs(data), {2, 2, 25, 42}, {0, 0, 0, 0, 1, 1, 2, 2});
  return t;
}
}  // namespace

TEST(VocabularyTreeIo, RoundTrip) {
  const auto t = small_tree();
  const auto p = tmp_path("voc_roundtrip.bin");
  t.save(p);
  const auto u = VocabularyTree::load(p);
  ASSERT_EQ(u.num_nodes(), t.num_nodes());
  ASSERT_EQ(u.num_words(), t.num_words());
  for (int i = 0; i < t.num_nodes(); ++i) {
    EXPECT_EQ(u.node(i).centroid, t.node(i).centroid);
    EXPECT_EQ(u.node(i).word, t.node(i).word);
    EXPECT_EQ(u.node(i).children, t.node(i).children);
  }
  for (int w = 0; w < t.num_words(); ++w) {
    EXPECT_EQ(u.word_node(w), t.word_node(w));
    EXPECT_FLOAT_EQ(u.idf(w), t.idf(w));
  }
  const std::vector<Desc256> img = {zeros(), with_low8(zeros(), true), ones()};
  EXPECT_EQ(u.bow(ptrs(img)), t.bow(ptrs(img)));
}

TEST(VocabularyTreeIo, EmptyInputTreeRoundTrips) {
  VocabularyTree t;
  t.train({}, {}, {});
  const auto p = tmp_path("voc_empty.bin");
  t.save(p);
  const auto u = VocabularyTree::load(p);
  EXPECT_EQ(u.num_nodes(), 1);
  EXPECT_EQ(u.num_words(), 0);
}

TEST(VocabularyTreeIo, BadFilesThrow) {
  EXPECT_THROW(VocabularyTree::load(tmp_path("does_not_exist.bin")), std::runtime_error);

  const auto p = tmp_path("voc_bad.bin");
  small_tree().save(p);
  std::string bytes;
  {
    std::ifstream in(p, std::ios::binary);
    bytes.assign(std::istreambuf_iterator<char>(in), {});
  }
  auto write_file = [&](const std::string& b) {
    std::ofstream(p, std::ios::binary | std::ios::trunc).write(b.data(), b.size());
  };

  write_file(bytes.substr(0, bytes.size() / 2));  // truncated
  EXPECT_THROW(VocabularyTree::load(p), std::runtime_error);

  auto bad_magic = bytes;
  bad_magic[0] = 'X';
  write_file(bad_magic);
  EXPECT_THROW(VocabularyTree::load(p), std::runtime_error);

  auto bad_version = bytes;
  bad_version[8] = 9;
  write_file(bad_version);
  EXPECT_THROW(VocabularyTree::load(p), std::runtime_error);
}

TEST(VocabularyTreeIo, SaveUntrainedThrows) {
  VocabularyTree t;
  EXPECT_THROW(t.save(tmp_path("voc_untrained.bin")), std::logic_error);
}