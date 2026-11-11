#pragma once
#include <random>
#include <vector>

#include "place_rec/binary_descriptor.hpp"

namespace pr::test {

// base with `flips` random bits flipped (a bit may be hit twice: fine)
inline Desc256 noisy(const Desc256& base, int flips, std::mt19937& gen) {
  Desc256 d = base;
  std::uniform_int_distribution<int> bit(0, 255);
  for (int f = 0; f < flips; ++f) {
    const int b = bit(gen);
    d[b / 64] ^= (1ULL << (b % 64));
  }
  return d;
}

// Call only AFTER `v` is fully built, or push_back may invalidate the pointers.
inline std::vector<const Desc256*> ptrs(const std::vector<Desc256>& v) {
  std::vector<const Desc256*> p;
  p.reserve(v.size());
  for (const auto& d : v) p.push_back(&d);
  return p;
}

inline Desc256 zeros() { return Desc256{}; }

inline Desc256 ones() {
  Desc256 d;
  d.fill(~0ULL);
  return d;
}

// set or clear the lowest 8 bits of word 0
inline Desc256 with_low8(Desc256 d, bool set) {
  if (set)
    d[0] |= 0xFFULL;
  else
    d[0] &= ~0xFFULL;
  return d;
}

}  // namespace pr::test