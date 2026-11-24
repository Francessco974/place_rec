// Tests for the match filters (abs_threshold, ratio_test, cross_check_filter).
// The filters only read d1, d2, cross_check, so rows are built by hand: no descriptors.

#include <gtest/gtest.h>

#include <algorithm>
#include <array>
#include <functional>
#include <vector>

#include "place_rec/matcher.hpp"

namespace pr {
namespace {

constexpr int kMaxDist = 50;
constexpr float kRatio = 0.8f;

MatchDescription row(int d1, int d2, bool cc = true) {
  MatchDescription m;
  m.query_index = 0;
  m.cand_index = 0;
  m.d1 = d1;
  m.d2 = d2;
  m.cross_check = cc;
  m.valid = true;
  return m;
}

std::vector<bool> valid_flags(const std::vector<MatchDescription>& rows) {
  std::vector<bool> v;
  for (const auto& m : rows) v.push_back(m.valid);
  return v;
}

// ---------------- abs_threshold ----------------

TEST(AbsThreshold, BoundaryIsInclusive) {
  std::vector<MatchDescription> rows = {row(49, 200), row(50, 200), row(51, 200)};
  FilterStats s;
  abs_threshold(rows, kMaxDist, s);
  EXPECT_EQ(valid_flags(rows), (std::vector<bool>{true, true, false}));
  EXPECT_EQ(s.rejected_by_dist, 1);
}

TEST(AbsThreshold, OnlyTouchesItsOwnCounter) {
  // Fails all three criteria, but only abs_threshold runs.
  std::vector<MatchDescription> rows = {row(100, 101, false)};
  FilterStats s;
  abs_threshold(rows, kMaxDist, s);
  EXPECT_FALSE(rows[0].valid);
  EXPECT_EQ(s.rejected_by_dist, 1);
  EXPECT_EQ(s.rejected_by_ratio, 0);
  EXPECT_EQ(s.rejected_by_cross, 0);
}

// ---------------- ratio_test ----------------

TEST(RatioTest, AcceptsDistinctiveMatches) {
  std::vector<MatchDescription> rows = {row(20, 90), row(0, 90)};
  FilterStats s;
  ratio_test(rows, kRatio, s);
  EXPECT_TRUE(rows[0].valid);
  EXPECT_TRUE(rows[1].valid);
  EXPECT_EQ(s.rejected_by_ratio, 0);
}

TEST(RatioTest, RejectsAmbiguousMatch) {
  std::vector<MatchDescription> rows = {row(40, 44)};
  FilterStats s;
  ratio_test(rows, kRatio, s);
  EXPECT_FALSE(rows[0].valid);
  EXPECT_EQ(s.rejected_by_ratio, 1);
}

// Two identical candidates: d1 == d2, including the perfect tie 0/0.
TEST(RatioTest, RejectsTies) {
  std::vector<MatchDescription> rows = {row(0, 0), row(30, 30)};
  FilterStats s;
  ratio_test(rows, kRatio, s);
  EXPECT_FALSE(rows[0].valid);
  EXPECT_FALSE(rows[1].valid);
  EXPECT_EQ(s.rejected_by_ratio, 2);
}

// Exact boundary with a ratio that is exact in float: 0.5 * 80 = 40.
TEST(RatioTest, BoundaryRejectsEquality) {
  std::vector<MatchDescription> rows = {row(39, 80), row(40, 80)};
  FilterStats s;
  ratio_test(rows, 0.5f, s);
  EXPECT_TRUE(rows[0].valid);
  EXPECT_FALSE(rows[1].valid);
}

// Single candidate: no second best, the test is skipped, valid is left as it was.
TEST(RatioTest, SkipsSingleCandidate) {
  std::vector<MatchDescription> rows = {row(20, k_no_dist), row(230, k_no_dist)};
  rows.push_back(row(20, k_no_dist));
  rows.back().valid = false;  // already rejected elsewhere: must stay false
  FilterStats s;
  ratio_test(rows, kRatio, s);
  EXPECT_EQ(valid_flags(rows), (std::vector<bool>{true, true, false}));
  EXPECT_EQ(s.rejected_by_ratio, 0);
}

// ---------------- cross_check_filter ----------------

TEST(CrossCheckFilter, RejectsNonMutual) {
  std::vector<MatchDescription> rows = {row(20, 90, true), row(20, 90, false)};
  FilterStats s;
  cross_check_filter(rows, s);
  EXPECT_EQ(valid_flags(rows), (std::vector<bool>{true, false}));
  EXPECT_EQ(s.rejected_by_cross, 1);
}

// ---------------- properties of the filter set ----------------

using Filter = std::function<void(std::vector<MatchDescription>&, FilterStats&)>;

std::array<Filter, 3> all_filters() {
  return {
      [](std::vector<MatchDescription>& r, FilterStats& s) { abs_threshold(r, kMaxDist, s); },
      [](std::vector<MatchDescription>& r, FilterStats& s) { ratio_test(r, kRatio, s); },
      [](std::vector<MatchDescription>& r, FilterStats& s) { cross_check_filter(r, s); },
  };
}

// One row per interesting combination.
std::vector<MatchDescription> mixed_rows() {
  return {
      row(20, 90, true),          // passes all
      row(60, 200, true),         // fails dist only
      row(40, 44, true),          // fails ratio only
      row(20, 90, false),         // fails cross only
      row(60, 62, true),          // fails dist + ratio
      row(60, 62, false),         // fails all three
      row(230, k_no_dist, true),  // single candidate: ratio skipped, fails dist
      row(10, k_no_dist, true),   // single candidate: passes
  };
}

const std::vector<bool> kExpectedValid = {true, false, false, false, false, false, false, true};

TEST(MatchFilters, OrderIndependent) {
  const auto filters = all_filters();
  std::array<int, 3> order = {0, 1, 2};
  int n_orders = 0;
  do {
    auto rows = mixed_rows();
    FilterStats s;
    for (int f : order) filters[f](rows, s);
    EXPECT_EQ(valid_flags(rows), kExpectedValid) << "order " << order[0] << order[1] << order[2];
    // Counts are independent of order too.
    EXPECT_EQ(s.rejected_by_dist, 4);
    EXPECT_EQ(s.rejected_by_ratio, 3);
    EXPECT_EQ(s.rejected_by_cross, 2);
    ++n_orders;
  } while (std::next_permutation(order.begin(), order.end()));
  EXPECT_EQ(n_orders, 6);
}

TEST(MatchFilters, CombinedEqualsIntersectionOfSingles) {
  const auto filters = all_filters();
  const std::size_t n = mixed_rows().size();
  std::vector<bool> intersection(n, true);
  for (const auto& f : filters) {
    auto rows = mixed_rows();
    FilterStats s;
    f(rows, s);
    for (std::size_t i = 0; i < n; ++i) intersection[i] = intersection[i] && rows[i].valid;
  }
  auto rows = mixed_rows();
  FilterStats s;
  for (const auto& f : filters) f(rows, s);
  EXPECT_EQ(valid_flags(rows), intersection);
}

// A row rejected by one filter is still counted by the others.
TEST(MatchFilters, StatsCountEveryRejection) {
  std::vector<MatchDescription> rows = {row(60, 62, false)};
  FilterStats s;
  for (const auto& f : all_filters()) f(rows, s);
  EXPECT_EQ(s.rejected_by_dist, 1);
  EXPECT_EQ(s.rejected_by_ratio, 1);
  EXPECT_EQ(s.rejected_by_cross, 1);
  EXPECT_EQ(count_valid(rows), 0);
}

TEST(MatchFilters, StatsAccumulateAcrossCalls) {
  std::vector<MatchDescription> rows = {row(60, 200)};
  FilterStats s;
  abs_threshold(rows, kMaxDist, s);
  abs_threshold(rows, kMaxDist, s);
  EXPECT_EQ(s.rejected_by_dist, 2);
}

// Filters never set valid back to true.
TEST(MatchFilters, NeverRevalidate) {
  std::vector<MatchDescription> rows = {row(20, 90, true)};
  rows[0].valid = false;
  FilterStats s;
  for (const auto& f : all_filters()) f(rows, s);
  EXPECT_FALSE(rows[0].valid);
}

TEST(MatchFilters, NoFiltersKeepsEverything) {
  const auto rows = mixed_rows();
  EXPECT_EQ(count_valid(rows), static_cast<int>(rows.size()));
}

TEST(MatchFilters, EmptyInputIsHarmless) {
  std::vector<MatchDescription> rows;
  FilterStats s;
  for (const auto& f : all_filters()) f(rows, s);
  EXPECT_EQ(count_valid(rows), 0);
  EXPECT_EQ(s.rejected_by_dist + s.rejected_by_ratio + s.rejected_by_cross, 0);
}

}  // namespace
}  // namespace pr