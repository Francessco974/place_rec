
#pragma once
#include <vector>

#include "place_rec/binary_descriptor.hpp"

namespace pr {
struct MatchDescription {
  int query_index{-1};
  int cand_index{-1};

  int d1{257};
  //   single-candidate behaviour d₂ stays 257
  int d2{257};
  bool cross_check{false};
  bool valid{false};
};

std::vector<MatchDescription> match_descriptors(const std::vector<Desc256>& query,
                                                const std::vector<Desc256>& candidate);
}  // namespace pr