#include "place_rec/matcher.hpp"

#include <cstddef>
#include <utility>
#include <vector>

#include "place_rec/binary_descriptor.hpp"

namespace pr {

std::vector<MatchDescription> match_descriptors(const std::vector<Desc256>& query,
                                                const std::vector<Desc256>& candidate) {
  std::vector<MatchDescription> bf_nn{};

  if (query.empty() || candidate.empty()) return bf_nn;
  std::vector<std::pair<int, int>> cross_checking;
  // index → candidate_d, <best_query_d, distance>
  cross_checking.assign(candidate.size(), {-1, 257});

  bf_nn.reserve(query.size());
  for (std::size_t i = 0; i < query.size(); i++) {
    MatchDescription curr_match{};
    curr_match.query_index = static_cast<int>(i);

    for (std::size_t j = 0; j < candidate.size(); j++) {
      const int curr_dist = hamming(query[i], candidate[j]);
      if (curr_dist < cross_checking[j].second) {
        cross_checking[j].second = curr_dist;
        cross_checking[j].first = static_cast<int>(i);
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
    m.cross_check = (cross_checking[m.cand_index].first == m.query_index);
  }

  return bf_nn;
}

}  // namespace pr