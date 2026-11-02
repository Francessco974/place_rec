#include "place_rec/binary_descriptor.hpp"

#include <opencv2/core/hal/interface.h>

#include <cstring>
#include <stdexcept>
#include <vector>

namespace pr {

std::vector<Desc256> pack(const cv::Mat& d) {
  if (d.empty()) return {};
  if (d.type() != CV_8U || d.cols != 32)
    throw std::invalid_argument("pack: expected N x 32 CV_8U descriptors");
  std::vector<Desc256> out(d.rows);
  for (int i = 0; i < d.rows; i++) std::memcpy(out[i].data(), d.ptr<uchar>(i), 32);
  return out;
}

Desc256 majority(const std::vector<const Desc256*>& members) {
  int counts[256] = {0};
  for (const Desc256* m : members)
    for (int w = 0; w < 4; ++w)
      for (int b = 0; b < 64; ++b) counts[w * 64 + b] += static_cast<int>(((*m)[w] >> b) & 1ULL);
  const int n = static_cast<int>(members.size());
  Desc256 c{};
  for (int w = 0; w < 4; ++w)
    for (int b = 0; b < 64; ++b)
      if (2 * counts[w * 64 + b] > n) c[w] |= (1ULL << b);
  return c;
}

}  // namespace pr