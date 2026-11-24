
#pragma once
#include <vector>

#include "place_rec/binary_descriptor.hpp"

namespace pr {

constexpr int k_no_dist{257};

struct MatchDescription {
  int query_index{-1};
  int cand_index{-1};

  int d1{k_no_dist};
  //   single-candidate behaviour d₂ stays k_no_dist
  int d2{k_no_dist};
  bool cross_check{false};
  bool valid{false};
};
// Each filter counts every row it rejects, so the counts can sum to more than the rows lost.
// Counters accumulate.
struct FilterStats {
  int rejected_by_dist{0};
  int rejected_by_ratio{0};
  int rejected_by_cross{0};
  int survivors{0};
};

// Allow custom filter
struct MatchFilterParams {
  bool use_max_dist{true};
  int max_dist{50};
  bool use_ratio{true};
  float ratio{0.8f};
  bool use_cross_check{true};
};

std::vector<MatchDescription> match_and_filter(const std::vector<Desc256>& query,
                                               const std::vector<Desc256>& candidate,
                                               const MatchFilterParams& conf, FilterStats& stats);

std::vector<MatchDescription> match_descriptors(const std::vector<Desc256>& query,
                                                const std::vector<Desc256>& candidate);
// A filter may only set valid to false, and decides from d1, d2, cross_check only
void abs_threshold(std::vector<MatchDescription>& matches, int threshold, FilterStats& stats);
void ratio_test(std::vector<MatchDescription>& matches, float ratio, FilterStats& stats);
void cross_check_filter(std::vector<MatchDescription>& matches, FilterStats& stats);

int count_valid(const std::vector<MatchDescription>& matches);

}  // namespace pr