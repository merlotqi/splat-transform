/***********************************************************************************
 *
 * splat - A C++ library for reading and writing 3D Gaussian Splatting (splat) files.
 *
 * This library provides functionality to convert, manipulate, and process
 * 3D Gaussian splatting data formats used in real-time neural rendering.
 *
 * This file is part of splat.
 *
 * splat is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 2 of the License, or
 * (at your option) any later version.
 *
 * splat is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this program; if not, write to the Free Software Foundation, Inc.,
 * 51 Franklin Street, Fifth Floor, Boston, MA 02110-1301, USA.
 *
 * For more information, visit the project's homepage or contact the author.
 *
 ***********************************************************************************/

#include <splat/maths/maths.h>
#include <splat/op/morton_order.h>
#include <splat/op/voxel_filter.h>

#include <cstddef>
#include <cstdint>
#include <unordered_map>
#include <unordered_set>
#include <vector>

namespace splat {

namespace details {

// ============================================================================
// Edge mask constants for 4x4x4 voxel blocks
// ============================================================================
// Bit layout: bitIdx = lx + ly*4 + lz*16
// lo = bits 0-31 (lz=0: 0-15, lz=1: 16-31)
// hi = bits 32-63 (lz=2: 0-15, lz=3: 16-31)

/** lx=0 positions in each 32-bit word */
static constexpr uint32_t FACE_X0 = 0x11111111;

/** lx=3 positions in each 32-bit word */
static constexpr uint32_t FACE_X3 = 0x88888888;

/** ly=0 positions in each 32-bit word */
static constexpr uint32_t FACE_Y0 = 0x000F000F;

/** ly=3 positions in each 32-bit word */
static constexpr uint32_t FACE_Y3 = 0xF000F000;

/** lz=0 positions: lo bits 0-15 */
static constexpr uint32_t FACE_Z0_LO = 0x0000FFFF;

/** lz=3 positions: hi bits 16-31 */
static constexpr uint32_t FACE_Z3_HI = 0xFFFF0000;

/**
 * Add cross-block face contribution for X/Y directions (shift stays within lo/hi words).
 *
 * @param nx - Adjacent block X coordinate.
 * @param ny - Adjacent block Y coordinate.
 * @param nz - Adjacent block Z coordinate.
 * @param solidSet - Set of Morton codes for solid blocks.
 * @param mixedMap - Map from Morton code to index in the mixed masks array.
 * @param masks - Interleaved voxel masks for mixed blocks [lo0, hi0, lo1, hi1, ...].
 * @param ourFaceMask - Bit mask selecting our block's face positions.
 * @param adjFaceMask - Bit mask selecting the adjacent block's opposite face positions.
 * @param shiftAmount - Number of bits to shift the adjacent face into our face positions.
 * @param shiftLeft - True to shift left, false to shift right.
 * @param curLo - Current low 32 bits of the direction mask.
 * @param curHi - Current high 32 bits of the direction mask.
 * @param write - Callback to write the updated (lo, hi) direction mask.
 */
static void addCrossFace(int nx, int ny, int nz, const std::unordered_set<uint32_t>& solidSet,
                         const std::unordered_map<uint32_t, size_t>& mixedMap, const std::vector<uint32_t>& masks,
                         uint32_t ourFaceMask, uint32_t adjFaceMask, int shiftAmount, bool shiftLeft, uint32_t curLo,
                         uint32_t curHi, std::function<void(uint32_t lo, uint32_t hi)> write) {
  const auto adjMorton = xyzToMorton(nx, ny, nz);

  if (solidSet.count(adjMorton)) {
    write(curLo | ourFaceMask, curHi | ourFaceMask);
    return;
  }

  if (auto it = mixedMap.find(adjMorton); it != mixedMap.end()) {
    const size_t adjIdx = it->second;
    const uint32_t adjLo = masks[adjIdx * 2];
    const uint32_t adjHi = masks[adjIdx * 2 + 1];
    const uint32_t faceLo = adjLo & adjFaceMask;
    const uint32_t faceHi = adjHi & adjFaceMask;

    if (shiftLeft) {
      write(curLo | (faceLo << shiftAmount), curHi | (faceHi << shiftAmount));
    } else {
      write(curLo | (faceLo >> shiftAmount), curHi | (faceHi >> shiftAmount));
    }
  } else {
    write(curLo, curHi);
  }
}

/**
 * Add cross-block face contribution for Z direction (crosses lo/hi boundary).
 *
 * +Z: our lz=3 (hi bits 16-31) <- adjacent's lz=0 (lo bits 0-15), shift left by 16
 * -Z: our lz=0 (lo bits 0-15) <- adjacent's lz=3 (hi bits 16-31), shift right by 16
 *
 * @param nx - Adjacent block X coordinate.
 * @param ny - Adjacent block Y coordinate.
 * @param nz - Adjacent block Z coordinate.
 * @param solidSet - Set of Morton codes for solid blocks.
 * @param mixedMap - Map from Morton code to index in the mixed masks array.
 * @param masks - Interleaved voxel masks for mixed blocks [lo0, hi0, lo1, hi1, ...].
 * @param plusZ - True for +Z direction, false for -Z.
 * @param curLo - Current low 32 bits of the direction mask.
 * @param curHi - Current high 32 bits of the direction mask.
 * @param write - Callback to write the updated (lo, hi) direction mask.
 */
static void addCrossFaceZ(int nx, int ny, int nz, const std::unordered_set<uint32_t>& solidSet,
                          const std::unordered_map<uint32_t, size_t>& mixedMap, const std::vector<uint32_t>& masks,
                          bool plusZ, uint32_t curLo, uint32_t curHi,
                          std::function<void(uint32_t lo, uint32_t hi)> write) {
  const uint32_t adjMorton = xyzToMorton(nx, ny, nz);

  if (solidSet.find(adjMorton) != solidSet.end()) {
    if (plusZ) {
      write(curLo, curHi | FACE_Z3_HI);
    } else {
      write(curLo | FACE_Z0_LO, curHi);
    }
    return;
  }

  if (auto it = mixedMap.find(adjMorton); it != mixedMap.end()) {
    const size_t adjIdx = it->second;
    const uint32_t adjLo = masks[adjIdx * 2];
    const uint32_t adjHi = masks[adjIdx * 2 + 1];

    if (plusZ) {
      write(curLo, curHi | ((adjLo & FACE_Z0_LO) << 16));
    } else {
      write(curLo | ((adjHi & FACE_Z3_HI) >> 16), curHi);
    }
  } else {
    write(curLo, curHi);
  }
}

}  // namespace details

BlockAccumulator filterAndFillBlocks(const BlockAccumulator& accumulator) {
  std::vector<uint32_t> mixedMorton = accumulator.mixedMorton;
  std::vector<uint32_t> solid = accumulator.solidMorton;
  std::vector<uint32_t> masks = accumulator.mixedMasks;

  // Build lookup structures from original (unmodified) data
  std::unordered_set<uint32_t> solidSet;
  for (size_t i = 0; i < accumulator.solidMorton.size(); ++i) {
    solidSet.insert(solid[i]);
  }

  std::unordered_map<uint32_t, size_t> mixedMap;
  for (size_t i = 0; i < mixedMorton.size(); ++i) {
    mixedMap.insert({mixedMorton[i], i});
  }

  // New masks array (snapshot: cross-block lookups always read the original masks)
  std::vector<int> newMasks(masks.size());
  int voxelsRemoved = 0;
  int voxelsFilled = 0;

  for (size_t i = 0; i < mixedMorton.size(); ++i) {
    const auto morton = mixedMorton[i];
    const auto origLo = masks[i * 2];
    const auto origHi = masks[i * 2 + 1];

    auto&& [bx, by, bz] = mortonToXYZ(morton);

    // --- In-block per-direction occupancy masks ---
    // Each gives: bit p = 1 iff position p's neighbor in that direction is occupied

    // +X: result[p] = mask[p+1], valid for lx < 3
    auto pxLo = (origLo >> 1) & ~details::FACE_X3;
    auto pxHi = (origHi >> 1) & ~details::FACE_X3;

    // -X: result[p] = mask[p-1], valid for lx > 0
    auto mxLo = (origLo << 1) & ~details::FACE_X0;
    auto mxHi = (origHi << 1) & ~details::FACE_X0;

    // +Y: result[p] = mask[p+4], valid for ly < 3
    auto pyLo = (origLo >> 4) & ~details::FACE_Y3;
    auto pyHi = (origHi >> 4) & ~details::FACE_Y3;

    // -Y: result[p] = mask[p-4], valid for ly > 0
    auto myLo = (origLo << 4) & ~details::FACE_Y0;
    auto myHi = (origHi << 4) & ~details::FACE_Y0;

    // +Z: result[p] = mask[p+16], crosses lo/hi at lz=1->lz=2
    auto pzLo = (origLo >> 16) | (origHi << 16);
    auto pzHi = origHi >> 16;

    // -Z: result[p] = mask[p-16], crosses lo/hi at lz=2->lz=1
    auto mzLo = origLo << 16;
    auto mzHi = (origHi << 16) | (origLo >> 16);

    // --- Cross-block contributions ---
    // For each face direction: look up the adjacent block, extract opposite-face
    // bits, shift to align with our face positions, OR into the direction mask.

    // +X: our lx=3 face <- adjacent's lx=0 face (shifted left by 3)
    details::addCrossFace(bx + 1, by, bz, solidSet, mixedMap, masks, details::FACE_X3, details::FACE_X0, 3, true, pxLo,
                          pxHi, [&](uint32_t lo, uint32_t hi) {
                            pxLo = lo;
                            pxHi = hi;
                          });

    // -X: our lx=0 face <- adjacent's lx=3 face (shifted right by 3)
    details::addCrossFace(bx - 1, by, bz, solidSet, mixedMap, masks, details::FACE_X0, details::FACE_X3, 3, false, mxLo,
                          mxHi, [&](uint32_t lo, uint32_t hi) {
                            mxLo = lo;
                            mxHi = hi;
                          });

    // +Y: our ly=3 face <- adjacent's ly=0 face (shifted left by 12)
    details::addCrossFace(bx, by + 1, bz, solidSet, mixedMap, masks, details::FACE_Y3, details::FACE_Y0, 12, true, pyLo,
                          pyHi, [&](uint32_t lo, uint32_t hi) {
                            pyLo = lo;
                            pyHi = hi;
                          });

    // -Y: our ly=0 face <- adjacent's ly=3 face (shifted right by 12)
    details::addCrossFace(bx, by - 1, bz, solidSet, mixedMap, masks, details::FACE_Y0, details::FACE_Y3, 12, false,
                          myLo, myHi, [&](uint32_t lo, uint32_t hi) {
                            myLo = lo;
                            myHi = hi;
                          });

    // +Z: our lz=3 face (hi bits 16-31) <- adjacent's lz=0 face (lo bits 0-15)
    details::addCrossFaceZ(bx, by, bz + 1, solidSet, mixedMap, masks, true, pzLo, pzHi, [&](uint32_t lo, uint32_t hi) {
      pzLo = lo;
      pzHi = hi;
    });

    // -Z: our lz=0 face (lo bits 0-15) <- adjacent's lz=3 face (hi bits 16-31)
    details::addCrossFaceZ(bx, by, bz - 1, solidSet, mixedMap, masks, false, mzLo, mzHi, [&](uint32_t lo, uint32_t hi) {
      mzLo = lo;
      mzHi = hi;
    });

    // --- Apply operations ---

    // Remove isolated voxels: keep only those with at least one occupied neighbor
    const auto neighborLo = pxLo | mxLo | pyLo | myLo | pzLo | mzLo;
    const auto neighborHi = pxHi | mxHi | pyHi | myHi | pzHi | mzHi;
    auto lo = origLo & neighborLo;
    auto hi = origHi & neighborHi;

    // Fill isolated empties: fill where all 6 neighbors are occupied
    const auto fillLo = ~lo & pxLo & mxLo & pyLo & myLo & pzLo & mzLo;
    const auto fillHi = ~hi & pxHi & mxHi & pyHi & myHi & pzHi & mzHi;
    lo |= fillLo;
    hi |= fillHi;

    voxelsRemoved += absl::popcount(origLo & ~lo) + absl::popcount(origHi & ~hi);
    voxelsFilled += absl::popcount(lo & ~origLo) + absl::popcount(hi & ~origHi);

    newMasks[i * 2] = lo;
    newMasks[i * 2 + 1] = hi;
  }

  return {};
}

}  // namespace splat
