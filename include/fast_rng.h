// Simple, fast RNG for host-side data generation
#pragma once

#include <cstdint>

namespace fast_rng {

struct FastRng {
  uint64_t s0, s1;
  static inline uint64_t splitmix64(uint64_t &x) {
    uint64_t z = (x += 0x9E3779B97F4A7C15ull);
    z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ull;
    z = (z ^ (z >> 27)) * 0x94D049BB133111EBull;
    return z ^ (z >> 31);
  }
  explicit FastRng(uint64_t seed) {
    uint64_t x = seed;
    s0 = splitmix64(x);
    s1 = splitmix64(x);
    if ((s0 | s1) == 0) s0 = 1; // avoid zero state
  }
  inline uint64_t next_u64() {
    uint64_t result = s0 + s1;
    s1 ^= s0;
    s0 = (s0 << 55) | (s0 >> (64 - 55));
    s0 ^= s1 ^ (s1 << 14);
    s1 = (s1 << 36) | (s1 >> (64 - 36));
    return result;
  }
  template <typename T> inline T next_signed();
};

template <> inline float FastRng::next_signed<float>() {
  uint64_t x = next_u64();
  float u = static_cast<float>((x >> 40) * (1.0 / 1099511627776.0)); // 2^40
  return 2.0f * u - 1.0f;
}

template <> inline double FastRng::next_signed<double>() {
  uint64_t x = next_u64();
  double u = (x >> 11) * (1.0 / 9007199254740992.0); // 2^53
  return 2.0 * u - 1.0;
}

} // namespace fast_rng
