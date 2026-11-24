// Tests for match_descriptors (brute-force NN + column bests for cross-check).
//
// Construction: 256 bits = 4 blocks of 64. Base descriptor = all zeros.
//   candidate c_j = base with block j flipped (j = 0, 1, 2)  -> candidates are 128 apart
//   query q       = c_target with k bits flipped in block 3   -> d(q, c_target) = k
//                                                                d(q, other c)  = 128 + k
// So every distance in these tests is known exactly.

#include <gtest/gtest.h>

#include <cstdint>
#include <limits>
#include <stdexcept>
#include <vector>

#include "place_rec/binary_descriptor.hpp"
#include "place_rec/matcher.hpp"

namespace pr {
namespace {

constexpr int kBlock = 64;
constexpr int kFreeBlockStart = 3 * kBlock;  // block 3: used only for query offsets

// ADAPT if Desc256 is not std::array<std::uint64_t, 4>.
// For std::bitset<256> the body is just: d.flip(bit);
void flip_bit(Desc256& d, int bit) { d[bit / 64] ^= (std::uint64_t{1} << (bit % 64)); }

void flip_bits(Desc256& d, int first, int count) {
  for (int b = first; b < first + count; ++b) flip_bit(d, b);
}

Desc256 make_candidate(int block) {
  Desc256 d{};
  flip_bits(d, block * kBlock, kBlock);
  return d;
}

// Query at distance k from candidate `target` (k <= 64).
Desc256 make_query(int target, int k) {
  Desc256 d = make_candidate(target);
  flip_bits(d, kFreeBlockStart, k);
  return d;
}

std::vector<Desc256> three_candidates() {
  return {make_candidate(0), make_candidate(1), make_candidate(2)};
}

// Sanity check of the construction itself: if this fails, every other test is meaningless.
TEST(MatcherFixture, ConstructionGivesKnownDistances) {
  const Desc256 q = make_query(1, 5);
  EXPECT_EQ(hamming(q, make_candidate(1)), 5);
  EXPECT_EQ(hamming(q, make_candidate(0)), 133);
  EXPECT_EQ(hamming(make_candidate(0), make_candidate(2)), 128);
}

TEST(MatchDescriptors, EmptyQueryReturnsEmpty) {
  EXPECT_TRUE(match_descriptors({}, three_candidates()).empty());
}

TEST(MatchDescriptors, EmptyCandidateReturnsEmpty) {
  EXPECT_TRUE(match_descriptors({make_query(0, 3)}, {}).empty());
}

TEST(MatchDescriptors, NearestAndSecondDistances) {
  const auto m = match_descriptors({make_query(1, 5)}, three_candidates());
  ASSERT_EQ(m.size(), 1u);
  EXPECT_EQ(m[0].query_index, 0);
  EXPECT_EQ(m[0].cand_index, 1);
  EXPECT_EQ(m[0].d1, 5);
  EXPECT_EQ(m[0].d2, 133);
  EXPECT_TRUE(m[0].valid);
  EXPECT_TRUE(m[0].cross_check);
}

// Regression for the d1/d2 bug: the second best arrives AFTER the best.
TEST(MatchDescriptors, SecondBestFoundAfterBest) {
  Desc256 near_dup = make_candidate(0);
  flip_bits(near_dup, 250, 6);  // c0 + 6 bits at the end of block 3
  const std::vector<Desc256> cands = {make_candidate(0), make_candidate(1), near_dup};
  // q flips bits 192..193 -> d(q,c0)=2, d(q,c1)=130, d(q,near_dup)=8
  const auto m = match_descriptors({make_query(0, 2)}, cands);
  ASSERT_EQ(m.size(), 1u);
  EXPECT_EQ(m[0].cand_index, 0);
  EXPECT_EQ(m[0].d1, 2);
  EXPECT_EQ(m[0].d2, 8);
}

// Decision: with one candidate there is no second best, d2 keeps the sentinel 257.
TEST(MatchDescriptors, SingleCandidateKeepsSentinelD2) {
  const auto m = match_descriptors({make_query(0, 3)}, {make_candidate(0)});
  ASSERT_EQ(m.size(), 1u);
  EXPECT_EQ(m[0].d1, 3);
  EXPECT_EQ(m[0].d2, 257);
}

// Two identical candidates: ambiguous, d1 == d2; strict '<' keeps the first.
TEST(MatchDescriptors, TieGivesEqualD1D2) {
  const auto m = match_descriptors({make_query(0, 4)}, {make_candidate(0), make_candidate(0)});
  ASSERT_EQ(m.size(), 1u);
  EXPECT_EQ(m[0].cand_index, 0);
  EXPECT_EQ(m[0].d1, 4);
  EXPECT_EQ(m[0].d2, 4);
}

// Many-to-one: both queries pick c0, but c0's best query is q0 -> only q0 passes.
TEST(MatchDescriptors, CrossCheckManyToOne) {
  const auto m = match_descriptors({make_query(0, 2), make_query(0, 7)}, three_candidates());
  ASSERT_EQ(m.size(), 2u);
  EXPECT_EQ(m[0].cand_index, 0);
  EXPECT_EQ(m[1].cand_index, 0);
  EXPECT_EQ(m[1].d1, 7);
  EXPECT_TRUE(m[0].cross_check);
  EXPECT_FALSE(m[1].cross_check);
}

TEST(MatchDescriptors, CrossCheckMutualPairs) {
  const auto m = match_descriptors({make_query(0, 3), make_query(2, 4)}, three_candidates());
  ASSERT_EQ(m.size(), 2u);
  EXPECT_EQ(m[0].cand_index, 0);
  EXPECT_EQ(m[1].cand_index, 2);
  EXPECT_TRUE(m[0].cross_check);
  EXPECT_TRUE(m[1].cross_check);
}

// The later query wins a column: cross-check must use the FINAL column bests.
TEST(MatchDescriptors, CrossCheckUsesFinalColumnBests) {
  const auto m = match_descriptors({make_query(0, 9), make_query(0, 1)}, three_candidates());
  ASSERT_EQ(m.size(), 2u);
  EXPECT_FALSE(m[0].cross_check);  // c0's best is q1 (1 < 9), found after q0 was processed
  EXPECT_TRUE(m[1].cross_check);
}

TEST(MatchDescriptors, ResultIsPerQueryInOrder) {
  const auto m =
      match_descriptors({make_query(2, 1), make_query(0, 1), make_query(1, 1)}, three_candidates());
  ASSERT_EQ(m.size(), 3u);
  for (int i = 0; i < 3; ++i) EXPECT_EQ(m[i].query_index, i);
  EXPECT_EQ(m[0].cand_index, 2);
  EXPECT_EQ(m[1].cand_index, 0);
  EXPECT_EQ(m[2].cand_index, 1);
}

TEST(MatchAndFilter, AllOffKeepsEveryRow) {
  MatchFilterParams p;
  p.use_max_dist = p.use_ratio = p.use_cross_check = false;
  FilterStats s;
  const auto m = match_and_filter({make_query(0, 2), make_query(0, 7)}, three_candidates(), p, s);
  EXPECT_EQ(count_valid(m), 2);
  EXPECT_EQ(s.survivors, 2);
  EXPECT_EQ(s.rejected_by_dist + s.rejected_by_ratio + s.rejected_by_cross, 0);
}

// q0: d1=2, mutual. q1: d1=7, loses c0 to q0. q2: d1=60 > 50.
TEST(MatchAndFilter, DefaultsReturnStatsAndAccumulate) {
  const std::vector<Desc256> q = {make_query(0, 2), make_query(0, 7), make_query(1, 60)};
  FilterStats s;
  const auto m = match_and_filter(q, three_candidates(), MatchFilterParams{}, s);
  EXPECT_TRUE(m[0].valid);
  EXPECT_FALSE(m[1].valid);
  EXPECT_FALSE(m[2].valid);
  EXPECT_EQ(s.rejected_by_dist, 1);
  EXPECT_EQ(s.rejected_by_ratio, 0);
  EXPECT_EQ(s.rejected_by_cross, 1);
  EXPECT_EQ(s.survivors, 1);
  match_and_filter(q, three_candidates(), MatchFilterParams{}, s);  // second call adds
  EXPECT_EQ(s.survivors, 2);
  EXPECT_EQ(s.rejected_by_dist, 2);
}

TEST(MatchAndFilter, InvalidParamsThrow) {
  FilterStats s;
  for (float r : {0.0f, 1.5f, std::numeric_limits<float>::quiet_NaN()}) {
    MatchFilterParams p;
    p.ratio = r;
    EXPECT_THROW(match_and_filter({}, {}, p, s), std::invalid_argument) << r;
  }
  for (int d : {-1, 257}) {
    MatchFilterParams p;
    p.max_dist = d;
    EXPECT_THROW(match_and_filter({}, {}, p, s), std::invalid_argument) << d;
  }
}

TEST(MatchAndFilter, DisabledFilterValueIsNotValidated) {
  MatchFilterParams p;
  p.use_ratio = false;
  p.ratio = 0.0f;
  FilterStats s;
  EXPECT_NO_THROW(match_and_filter({}, {}, p, s));
}

}  // namespace
}  // namespace pr