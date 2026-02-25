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

#include <splat/models/data-table.h>
#include <splat/spatial/sparse_octree.h>

#include <cstddef>
#include <cstdint>
#include <vector>

namespace splat {

namespace block {

static constexpr int empty = 0;
static constexpr int solid = 1;
static constexpr int mixed = 2;

}  // namespace block

/**
 * Per-level data stored during bottom-up construction.
 * Uses Structure-of-Arrays layout to avoid per-node object allocation.
 */
struct LevelData {
  /** Sorted Morton codes for nodes at this level */
  std::vector<uint32_t> mortons;

  /** Block type for each node (Solid or Mixed) */
  std::vector<int> types;

  /** For level-0 Mixed nodes: index into mixed.masks. Otherwise -1. */
  std::vector<int32_t> maskIndices;

  /** For interior nodes (Mixed at level > 0): 8-bit child presence mask */
  std::vector<uint8_t> childMasks;
};

/**
 * Flatten the level-based tree into Laine-Karras format arrays using
 * wave-based BFS traversal from root down through levels.
 *
 * Uses parallel arrays for BFS waves (no per-node object allocation)
 * and binary search on sorted level mortons to locate children.
 *
 * @param levels - Array of per-level SoA data (index 0 = leaves, last = root).
 * @param mixedMasks - Interleaved voxel masks for mixed leaf blocks.
 * @param gridBounds - Grid bounds aligned to block boundaries.
 * @param sceneBounds - Original Gaussian scene bounds.
 * @param voxelResolution - Size of each voxel in world units.
 * @param treeDepth - Maximum tree depth.
 * @returns Sparse octree structure in Laine-Karras format.
 */
static SparseOctree flattenTreeFromLevels(const std::vector<LevelData>& levels, const std::vector<uint32_t> mixedMasks,
                                          const Bounds& gridBounds, const Bounds& sceneBounds, double voxelResolution,
                                          size_t treeDepth) {
  const LevelData& rootLevel = levels.back();

  if (rootLevel.mortons.size() == 0) {
    SparseOctree oct;
    oct.gridBounds = gridBounds;
    oct.sceneBounds = sceneBounds;
    oct.voxelResolution = voxelResolution;
    oct.leafSize = 4;
    oct.treeDepth = treeDepth;
    oct.numInteriorNodes = 0;
    oct.numMixedLeaves = 0;
    return oct;
  }

  // Upper bound on total nodes (not all may be reachable if solids collapsed)
  size_t maxNodes = 0;
  for (auto&& level : levels) {
    maxNodes += level.mortons.size();
  }

  std::vector<uint32_t> nodes(maxNodes);
  std::vector<uint32_t> leafDataList;
  uint32_t numInteriorNodes = 0;
  uint32_t numMixedLeaves = 0;
  uint32_t emitPos = 0;

  // BFS wave as parallel arrays (avoids object allocation per queue entry)
  std::vector<size_t> waveLi;
  std::vector<size_t> waveIi;

  // Initialize wave with root level entries
  size_t rootLi = levels.size() - 1;
  for (size_t i = 0; i < rootLevel.mortons.size(); ++i) {
    waveLi.push_back(rootLi);
    waveIi.push_back(i);
  }

  // Reusable arrays for tracking interior nodes within each wave
  std::vector<uint32_t> intPos;
  std::vector<uint32_t> intLi;
  std::vector<uint32_t> intIi;
  std::vector<uint32_t> intMask;

  // Reusable arrays for tracking interior nodes within each wave
  while (waveLi.size() > 0) {
    // Clear interior tracking arrays
    intPos.clear();
    intLi.clear();
    intIi.clear();
    intMask.clear();

    // Emit all nodes in this wave
    for (size_t w = 0; w < waveLi.size(); w++) {
      auto li = waveLi[w];
      auto ii = waveIi[w];
      auto level = levels[li];
      auto type = level.types[ii];

      // A node is a leaf if it's Solid (at any level, collapsed or original)
      // or if it's at level 0 (the leaf block level).
      auto isLeaf = (type == block::solid) || (li == 0);

      if (isLeaf) {
        if (type == block::solid) {
          nodes[emitPos] = SOLID_LEAF_MARKER;
        } else {
          // Mixed leaf — store index into leafData
          auto maskIdx = level.maskIndices[ii];
          auto leafDataIndex = leafDataList.size() >> 1;
          leafDataList.push_back(mixedMasks[maskIdx * 2]);
          leafDataList.push_back(mixedMasks[maskIdx * 2 + 1]);
          nodes[emitPos] = leafDataIndex & 0x00FFFFFF;
          numMixedLeaves++;
        }
      } else {
        // Interior node — record position for backfill after wave
        intPos.push_back(emitPos);
        intLi.push_back(li);
        intIi.push_back(ii);
        intMask.push_back(level.childMasks[ii]);
        numInteriorNodes++;
        // Placeholder (will be filled below)
        nodes[emitPos] = 0;
      }
      emitPos++;
    }

    // Build next wave from children of interior nodes.
    // Backfill interior node encodings with correct baseOffset.
    std::vector<size_t> nextWaveLi;
    std::vector<size_t> nextWaveIi;
    auto nextChildStart = emitPos;

    for (size_t j = 0; j < intPos.size(); j++) {
      auto childMask = intMask[j];
      auto childCount = absl::popcount(childMask);

      // Encode interior node: mask in high byte, baseOffset in low 24 bits
      nodes[intPos[j]] = ((childMask & 0xFF) << 24) | (nextChildStart & 0x00FFFFFF);

      // Find children in the level below using binary search.
      // Since each level's mortons are sorted, this is O(log n) per lookup.
      auto childLi = intLi[j] - 1;
      auto childLevel = levels[childLi];
      auto myMorton = levels[intLi[j]].mortons[intIi[j]];
      auto childMortonBase = myMorton * 8;
      auto childMortonEnd = childMortonBase + 8;
      auto childMortons = childLevel.mortons;

      // Binary search for first child with morton >= childMortonBase
      auto lo = 0;
      auto hi = childMortons.size();
      while (lo < hi) {
        auto mid = (lo + hi) >> 1;
        if (childMortons[mid] < childMortonBase)
          lo = mid + 1;
        else
          hi = mid;
      }

      // Collect all children in morton order (they are contiguous in sorted array)
      while (lo < childMortons.size() && childMortons[lo] < childMortonEnd) {
        nextWaveLi.push_back(childLi);
        nextWaveIi.push_back(lo);
        lo++;
      }

      nextChildStart += childCount;
    }

    waveLi = nextWaveLi;
    waveIi = nextWaveIi;
  }

  return {};
}

SparseOctree buildSparseOctree(const BlockAccumulator& accumulator, const Bounds& gridBounds, const Bounds& sceneBounds,
                               double voxelResolution) {
  return {};
}

}  // namespace splat
