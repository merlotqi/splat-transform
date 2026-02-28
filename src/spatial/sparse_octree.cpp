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
#include <splat/models/data-table.h>
#include <splat/spatial/sparse_octree.h>
#include <sys/types.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <numeric>
#include <vector>

namespace splat {

namespace block {

static constexpr uint8_t empty = 0;
static constexpr uint8_t solid = 1;
static constexpr uint8_t mixed = 2;

}  // namespace block

/**
 * Per-level data stored during bottom-up construction.
 * Uses Structure-of-Arrays layout to avoid per-node object allocation.
 */
struct LevelData {
  /** Sorted Morton codes for nodes at this level */
  std::vector<uint32_t> mortons;

  /** Block type for each node (Solid or Mixed) */
  std::vector<uint8_t> types;

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
                                          const Bounds& gridBounds, const Bounds& sceneBounds, float voxelResolution,
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

  SparseOctree oct;
  oct.gridBounds = gridBounds;
  oct.sceneBounds = sceneBounds;
  oct.voxelResolution = voxelResolution;
  oct.leafSize = 4;
  oct.treeDepth = treeDepth;
  oct.numInteriorNodes = numInteriorNodes;
  oct.numMixedLeaves = numMixedLeaves;
  if (maxNodes) {
    oct.nodes = nodes;
  } else {
    oct.nodes.insert(oct.nodes.begin(), nodes.begin(), nodes.begin() + emitPos);
  }
  oct.leafData = leafDataList;

  return oct;
}

SparseOctree buildSparseOctree(const BlockAccumulator& accumulator, const Bounds& gridBounds, const Bounds& sceneBounds,
                               float voxelResolution) {
  auto tProfile = std::chrono::high_resolution_clock::now();

  auto mixedmorton = accumulator.mixedMorton;
  auto solid = accumulator.solidMorton;
  const size_t totalBlocks = mixedmorton.size() + solid.size();

  // --- Phase 1: Combine blocks into SoA arrays and sort by Morton code ---
  // Avoids creating per-block objects (BlockEntry) — uses parallel arrays.

  std::vector<uint32_t> mortons(totalBlocks);
  std::vector<uint32_t> types(totalBlocks);
  std::vector<int32_t> maskIndices(totalBlocks);

  size_t idx = 0;
  for (int32_t i = 0; i < (int32_t)mixedmorton.size(); ++i) {
    mortons[idx] = mixedmorton[i];
    types[idx] = block::mixed;
    maskIndices[idx] = i;
    idx++;
  }

  for (int32_t i = 0; i < (int32_t)solid.size(); ++i) {
    mortons[idx] = solid[i];
    types[idx] = block::solid;
    maskIndices[idx] = -1;
    idx++;
  }
  // Co-sort by Morton code using an index permutation array
  std::vector<size_t> sortOrder(totalBlocks);
  std::iota(sortOrder.begin(), sortOrder.end(), 0);
  std::sort(sortOrder.begin(), sortOrder.end(), [&](size_t a, size_t b) { return mortons[a] - mortons[b]; });

  std::vector<uint32_t> sortedMortons(totalBlocks);
  std::vector<uint8_t> sortedTypes(totalBlocks);
  std::vector<int32_t> sortedMaskIndices(totalBlocks);
  for (size_t i = 0; i < totalBlocks; ++i) {
    const size_t si = sortOrder[i];
    sortedMortons[i] = mortons[si];
    sortedTypes[i] = types[si];
    sortedMaskIndices[i] = maskIndices[si];
  }

  const auto tSort = std::chrono::high_resolution_clock::now();

  // --- Phase 2: Build tree bottom-up level by level using linear scan ---
  // Instead of Map<number, BuildNode> per level, we use sorted parallel
  // arrays and exploit the fact that sorted Morton codes make parent
  // grouping a simple linear scan (consecutive entries with the same
  // floor(morton/8) share a parent).

  // Calculate tree depth based on grid size
  const Eigen::Vector3f gridSize = gridBounds.max - gridBounds.min;
  const float blockSize = voxelResolution * 4.0f;
  const float blocksPerSize = maxs(std::ceil(gridSize.x() / blockSize), std::ceil(gridSize.y() / blockSize),
                                   std::ceil(gridSize.z() / blockSize));
  const int treeDepth = std::max(1, (int)std::ceil(std::log2(blocksPerSize)));

  // Store level data for each tree level (level 0 = leaves, higher = toward root)
  std::vector<LevelData> levels;

  // Current level data starts as the sorted leaf blocks
  auto curMortons = sortedMortons;
  auto curTypes = sortedTypes;
  auto curMaskIndices = sortedMaskIndices;
  // Leaf level has no child masks (leaves have no children)
  std::vector<uint8_t> curChildMasks(totalBlocks, 0);

  // Build up level by level
  int actualDepth = treeDepth;
  int levelSteps = 8;

  for (int level = 0; level < treeDepth; ++level) {
    // Save current level before building the next one above
    levels.push_back({curMortons, curTypes, curMaskIndices, curChildMasks});

    // Build next level using linear scan on sorted data.
    // Since curMortons is sorted, entries sharing the same parent
    // (floor(morton/8)) are contiguous — no Map needed.
    size_t n = curMortons.size();
    std::vector<uint32_t> nextMortons;
    std::vector<uint8_t> nextTypes;
    std::vector<int32_t> nextMaskIndices;
    std::vector<uint8_t> nextChildMasks;

    size_t i = 0;
    while (i < n) {
      auto parentMorton = std::floor(curMortons[i] / 8);
      int childMask = 0;
      bool allSolid = true;
      int childCount = 0;

      // Scan all consecutive entries that share this parent
      while (i < n && std::floor(curMortons[i] / 8) == parentMorton) {
        auto octant = curMortons[i] % 8;
        childMask |= (1 << octant);
        if (curTypes[i] != block::solid) {
          allSolid = false;
        }
        childCount++;
        i++;
      }

      if (allSolid && childCount == 8) {
        // All 8 children are solid — collapse to solid parent
        nextMortons.push_back(parentMorton);
        nextTypes.push_back(block::solid);
        nextMaskIndices.push_back(-1);
        nextChildMasks.push_back(0);
      } else {
        // Interior node with sparse children
        nextMortons.push_back(parentMorton);
        nextTypes.push_back(block::mixed);
        nextMaskIndices.push_back(-1);
        nextChildMasks.push_back(childMask);
      }
    }

    curMortons = nextMortons;
    curTypes = nextTypes;
    curMaskIndices = nextMaskIndices;
    curChildMasks = nextChildMasks;

    // Break when the tree is empty or has converged to a single root at Morton 0.
    // We must NOT break early if the single remaining node has a non-zero Morton,
    // because the reader reconstructs Morton codes starting from root Morton 0.
    if (curMortons.size() == 0 || (curMortons.size() == 1 && curMortons[0] == 0)) {
      actualDepth = level + 1;
      break;
    }
  }

  // Save the root level
  levels.push_back({curMortons, curTypes, curMaskIndices, curChildMasks});

  auto tBuild = std::chrono::high_resolution_clock::now();

  // --- Phase 3: Flatten tree to Laine-Karras format ---
  // Uses wave-based BFS on level arrays, avoiding BuildNode objects
  // and the O(n²) queue.shift() of the original approach.
  SparseOctree result =
      flattenTreeFromLevels(levels, accumulator.mixedMasks, gridBounds, sceneBounds, voxelResolution, actualDepth);

  auto tFlattern = std::chrono::high_resolution_clock::now();

  return result;
}

}  // namespace splat
