#include "place_rec/matcher.hpp"

#include <cstddef>
#include <vector>

#include "place_rec/binary_descriptor.hpp"

namespace pr {

namespace {
struct ColumnBest {
  int query_index{-1};
  int dist{k_no_dist};
};
}  // namespace

std::vector<MatchDescription> match_descriptors(const std::vector<Desc256>& query,
                                                const std::vector<Desc256>& candidate) {
  std::vector<MatchDescription> bf_nn{};

  if (query.empty() || candidate.empty()) return bf_nn;

  std::vector<ColumnBest> col_best(candidate.size());
  bf_nn.reserve(query.size());
  for (std::size_t i = 0; i < query.size(); i++) {
    MatchDescription curr_match{};
    curr_match.query_index = static_cast<int>(i);

    for (std::size_t j = 0; j < candidate.size(); j++) {
      const int curr_dist = hamming(query[i], candidate[j]);
      if (curr_dist < col_best[j].dist) {
        col_best[j].dist = curr_dist;
        col_best[j].query_index = static_cast<int>(i);
      }

      if (curr_dist < curr_match.d1) {
        curr_match.d2 = curr_match.d1;
        curr_match.d1 = curr_dist;
        curr_match.cand_index = static_cast<int>(j);
        curr_match.valid = true;
      } else if (curr_dist < curr_match.d2) {
        curr_match.d2 = curr_dist;
      }
    }

    bf_nn.push_back(curr_match);
  }

  for (auto& m : bf_nn) {
    m.cross_check = (col_best[m.cand_index].query_index == m.query_index);
  }

  return bf_nn;
}

void abs_threshold(std::vector<MatchDescription>& matches, int threshold, FilterStats& stats) {
  for (auto& m : matches) {
    if (m.d1 > threshold) {
      m.valid = false;
      stats.rejected_by_dist += 1;
    }
  }
}
void ratio_test(std::vector<MatchDescription>& matches, float ratio, FilterStats& stats) {
  for (auto& m : matches) {
    if (m.d2 == k_no_dist) continue;
    const float d1 = static_cast<float>(m.d1);
    const float d2 = static_cast<float>(m.d2);
    if (d1 >= ratio * d2) {
      m.valid = false;
      stats.rejected_by_ratio += 1;
    }
  }
}
void cross_check_filter(std::vector<MatchDescription>& matches, FilterStats& stats) {
  for (auto& m : matches) {
    if (!m.cross_check) {
      m.valid = false;
      stats.rejected_by_cross += 1;
    }
  }
}

int count_valid(const std::vector<MatchDescription>& matches) {
  return static_cast<int>(std::count_if(matches.begin(), matches.end(),
                                        [](const MatchDescription& m) { return m.valid; }));
}

}  // namespace pr