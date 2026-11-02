#pragma once

#include <array>
#include <cstdint>
#include <opencv2/core/mat.hpp>
#include <vector>

namespace pr {

using Desc256 = std::array<std::uint64_t, 4>;
inline int hamming(const Desc256& a, const Desc256& b) {
  return __builtin_popcountll(a[0] ^ b[0]) + __builtin_popcountll(a[1] ^ b[1]) +
         __builtin_popcountll(a[2] ^ b[2]) + __builtin_popcountll(a[3] ^ b[3]);
}

std::vector<Desc256> pack(const cv::Mat& descriptors);
Desc256 majority(const std::vector<const Desc256*>& members);

}  // namespace pr