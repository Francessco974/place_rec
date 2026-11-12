#include <gtest/gtest.h>

#include <vector>

#include "place_rec/vocabulary_tree.hpp"
#include "test_helpers.hpp"

using pr::Desc256;
using pr::VocabularyTree;
using namespace pr::test;
using BV = VocabularyTree::BowVector;

// ---------- bow() ----------

TEST(VocabularyTreeBow, SumsToOneAndIsSorted) {
  const auto a1 = zeros(), a2 = with_low8(zeros(), true);
  const auto b1 = ones(), b2 = with_low8(ones(), false);
  std::vector<Desc256> data = {a1, a1, a2, a2, a1, b1, b2, b2};
  VocabularyTree t;
  t.train(ptrs(data), {2, 2, 25, 42}, {0, 0, 0, 0, 1, 1, 2, 2});
  const std::vector<Desc256> img = {a1, a1, a2, b1};
  const auto v = t.bow(ptrs(img));
  ASSERT_FALSE(v.empty());
  float s = 0;
  for (size_t i = 0; i < v.size(); ++i) {
    s += v[i].second;
    EXPECT_GT(v[i].second, 0.0f);
    if (i) EXPECT_LT(v[i - 1].first, v[i].first);
  }
  EXPECT_NEAR(s, 1.0f, 1e-5f);
  EXPECT_EQ(v, t.bow(ptrs(img)));  // deterministic
}

TEST(VocabularyTreeBow, EmptyCases) {
  VocabularyTree untrained;
  const std::vector<Desc256> img = {zeros()};
  EXPECT_TRUE(untrained.bow(ptrs(img)).empty());
  EXPECT_TRUE(untrained.bow({}).empty());
}

// ---------- similarity_score() ----------

TEST(SimilarityScore, IdenticalIsOne) {
  const BV v = {{2, 0.5f}, {7, 0.5f}};
  EXPECT_NEAR(pr::similarity_score(v, v), 1.0f, 1e-6f);
}

TEST(SimilarityScore, ToyExample) {
  const BV v = {{2, 0.5f}, {7, 0.5f}};
  const BV w = {{2, 0.25f}, {5, 0.5f}, {7, 0.25f}};
  EXPECT_NEAR(pr::similarity_score(v, w), 0.5f, 1e-6f);
  EXPECT_NEAR(pr::similarity_score(w, v), 0.5f, 1e-6f);  // symmetric
}

TEST(SimilarityScore, DisjointAndEmptyGiveZero) {
  const BV v = {{1, 1.0f}};
  const BV w = {{2, 1.0f}};
  EXPECT_FLOAT_EQ(pr::similarity_score(v, w), 0.0f);
  EXPECT_FLOAT_EQ(pr::similarity_score(v, BV{}), 0.0f);
  EXPECT_FLOAT_EQ(pr::similarity_score(BV{}, BV{}), 0.0f);
}

TEST(SimilarityScore, MatchesL1Formula) {
  const BV v = {{0, 0.2f}, {3, 0.3f}, {9, 0.5f}};
  const BV w = {{3, 0.6f}, {4, 0.1f}, {9, 0.3f}};
  // 1 - 0.5 * sum|v-w| over the union of words
  const float l1 = 0.2f + 0.3f + 0.1f + 0.2f;  // words 0, 3, 4, 9
  EXPECT_NEAR(pr::similarity_score(v, w), 1.0f - 0.5f * l1, 1e-6f);
}

#include <opencv2/core.hpp>

#include "place_rec/bow_recognizer.hpp"

namespace {
cv::Mat to_mat(const std::vector<pr::Desc256>& v) {
  cv::Mat m(static_cast<int>(v.size()), 32, CV_8U);
  for (size_t i = 0; i < v.size(); ++i) std::memcpy(m.ptr<uchar>(i), v[i].data(), 32);
  return m;
}
pr::Keyframe make_kf(int id, double stamp, const std::vector<pr::Desc256>& d) {
  pr::Keyframe kf;
  kf.id = id;
  kf.stamp = stamp;
  kf.descriptors = to_mat(d);
  return kf;
}
std::shared_ptr<const pr::VocabularyTree> small_vocab() {
  const auto a1 = zeros(), a2 = with_low8(zeros(), true);
  const auto b1 = ones(), b2 = with_low8(ones(), false);
  static std::vector<pr::Desc256> data = {a1, a1, a2, a2, a1, b1, b2, b2};
  auto t = std::make_shared<pr::VocabularyTree>();
  t->train(ptrs(data), {2, 2, 25, 42}, {0, 0, 0, 0, 1, 1, 2, 2});
  return t;
}
}  // namespace

TEST(BowRecognizer, NullVocabularyThrows) {
  EXPECT_THROW(pr::BowRecognizer(nullptr), std::invalid_argument);
}

TEST(BowRecognizer, NonSequentialIdThrows) {
  pr::BowRecognizer rec(small_vocab());
  EXPECT_THROW(rec.add(make_kf(3, 0.0, {zeros()})), std::logic_error);
}

TEST(BowRecognizer, SelfMatchWhenWindowIncludesIt) {
  pr::BowRecognizer rec(small_vocab());
  const std::vector<pr::Desc256> d = {zeros(), with_low8(zeros(), true), ones()};
  rec.add(make_kf(0, 1.0, d));
  const auto r =
      rec.query(make_kf(1, 2.0, d), /*max_stamp=*/1.0, 5);  // stamp == max_stamp: included
  ASSERT_EQ(r.size(), 1u);
  EXPECT_EQ(r[0].id, 0);
  EXPECT_NEAR(r[0].score, 1.0, 1e-5);
}

TEST(BowRecognizer, WindowExcludesRecent) {
  pr::BowRecognizer rec(small_vocab());
  const std::vector<pr::Desc256> d = {zeros(), ones()};
  rec.add(make_kf(0, 1.0, d));
  EXPECT_TRUE(rec.query(make_kf(1, 2.0, d), 0.5, 5).empty());
}

TEST(BowRecognizer, NoZeroScoresAndSortedAndCapped) {
  pr::BowRecognizer rec(small_vocab());
  for (int i = 0; i < 6; ++i) rec.add(make_kf(i, i, {i % 2 ? ones() : zeros()}));
  const auto r = rec.query(make_kf(6, 10.0, {zeros()}), 10.0, 3);
  EXPECT_LE(r.size(), 3u);
  for (size_t i = 0; i < r.size(); ++i) {
    EXPECT_GT(r[i].score, 0.0);
    if (i) EXPECT_GE(r[i - 1].score, r[i].score);
  }
}

TEST(BowRecognizer, EmptyDatabaseAndBadMaxResults) {
  pr::BowRecognizer rec(small_vocab());
  EXPECT_TRUE(rec.query(make_kf(0, 1.0, {zeros()}), 1.0, 5).empty());
  rec.add(make_kf(0, 0.0, {zeros()}));
  EXPECT_TRUE(rec.query(make_kf(1, 1.0, {zeros()}), 1.0, 0).empty());
}