#pragma once

#include <array>
#include <cstdint>
#include <math.h>
#include <string>
#include <vector>
#include <optional>

#ifdef __SSE2__
#include <emmintrin.h>
#include <xmmintrin.h>
#endif

namespace reader {
namespace sog {

struct Meta {
  int version;
  int count;

  struct {
    std::vector<double> mins;
    std::vector<double> maxs;
    std::vector<std::string> files;
  } means;

  struct {
    std::vector<double> codebook;
    std::vector<std::string> files;
  } scales;

  struct {
    std::vector<std::string> files;
  } quates;

  struct {
    std::vector<double> codebook;
    std::vector<std::string> files;
  } sh0;

  struct SHN {
    int count;
    int bands;
    std::vector<double> codebook;
    std::vector<std::string> files;
  };

  std::optional<SHN> shN;
};



static inline double sigmoidInv(double y) {
  const double e = std::min(1 - 1e-6, std::max(1e-6, y));
  return log(e / (1 - e));
}

static double invLogTransform(double v) {
  const double a = abs(v);
  const double e = exp(a) - 1;
  return v < 0 ? -e : e;
}

static inline std::array<float, 4> unpackQuat(uint8_t px, uint8_t py,
                                              uint8_t pz, uint8_t tag) {
  const uint8_t maxComp = tag - 252;
  const float a = static_cast<float>(px) / 255.0f * 2.0f - 1.0f;
  const float b = static_cast<float>(py) / 255.0f * 2.0f - 1.0f;
  const float c = static_cast<float>(pz) / 255.0f * 2.0f - 1.0f;
  constexpr float sqrt2 = 1.41421356237f;
  std::array<float, 4> comps = {0.0f, 0.0f, 0.0f, 0.0f};
  static constexpr std::array<std::array<uint8_t, 3>, 4> idx = {
      {{{1, 2, 3}}, {{0, 2, 3}}, {{0, 1, 3}}, {{0, 1, 2}}}};

  const auto &indices = idx[maxComp];
  comps[indices[0]] = a / sqrt2;
  comps[indices[1]] = b / sqrt2;
  comps[indices[2]] = c / sqrt2;

  float t = 1.0f - (comps[0] * comps[0] + comps[1] * comps[1] +
                    comps[2] * comps[2] + comps[3] * comps[3]);
  comps[maxComp] = sqrt(std::max(0.0f, t));

  return comps;
}

void read_sog();

} // namespace sog
} // namespace reader
