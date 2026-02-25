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

#pragma once

#include <splat/maths/maths.h>

#include <Eigen/Dense>
#include <vector>

namespace splat {

class DataTable;

struct Bounds {
  Eigen::Vector3d min;
  Eigen::Vector3d max;
};

struct BlockAccumulator {
  /** Morton codes for mixed blocks */
  std::vector<uint32_t> mixedMorton;
  /** Interleaved voxel masks for mixed blocks: [lo0, hi0, lo1, hi1, ...] */
  std::vector<uint32_t> mixedMasks;
  /** Morton codes for solid blocks (mask is implicitly all 1s) */
  std::vector<uint32_t> solidMorton;

 public:
  /**
   * Add a non-empty block to the accumulator.
   * Automatically classifies as solid or mixed based on mask values.
   *
   * @param morton - Morton code encoding block position
   * @param lo - Lower 32 bits of voxel mask
   * @param hi - Upper 32 bits of voxel mask
   */
  inline void addBlock(uint32_t morton, uint32_t lo, uint32_t hi) {
    if (isEmpty(lo, hi)) {
      return;
    }
    if (isSolid(lo, hi)) {
      solidMorton.push_back(morton);
    } else {
      mixedMorton.push_back(morton);
      mixedMasks.push_back(lo);
      mixedMasks.push_back(hi);
    }
  }

  inline size_t count() const { return mixedMorton.size() + solidMorton.size(); }
  inline void clear() {
    mixedMorton.clear();
    solidMorton.clear();
    mixedMasks.clear();
  }
};

/**
 * Sparse voxel octree using Laine-Karras node format.
 */
struct SparseOctree {
  /** Grid bounds aligned to 4x4x4 block boundaries */
  Bounds gridBounds;

  /** Original Gaussian scene bounds */
  Bounds sceneBounds;

  /** Size of each voxel in world units */
  double voxelResolution{0.0};

  /** Voxels per leaf dimension (always 4) */
  int leafSize{0};

  /** Maximum tree depth */
  int treeDepth{0};

  /** Number of interior nodes */
  size_t numInteriorNodes{0};

  /** Number of mixed leaf nodes */
  size_t numMixedLeaves{0};

  /** All nodes in Laine-Karras format (interior + leaves) */
  std::vector<uint32_t> nodes;

  /** Voxel masks for mixed leaves: pairs of u32 (lo, hi) */
  std::vector<uint32_t> leafData;
};

/**
 * Build a sparse octree from accumulated voxelization blocks.
 *
 * Uses Structure-of-Arrays (SoA) representation and linear scans on sorted
 * Morton codes instead of Maps and per-node objects for performance.
 *
 * @param accumulator - BlockAccumulator containing voxelized blocks
 * @param gridBounds - Grid bounds aligned to block boundaries
 * @param sceneBounds - Original scene bounds
 * @param voxelResolution - Size of each voxel in world units
 * @returns Sparse octree structure
 */
SparseOctree buildSparseOctree(const BlockAccumulator& accumulator, const Bounds& gridBounds, const Bounds& sceneBounds,
                               double voxelResolution);

}  // namespace splat
