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

#include <absl/types/span.h>

#include <memory>
#include <vector>

namespace splat {

class DataTable;

/**
 * @brief Bounding Volume Hierarchy (BVH) tree implementation using Axis-Aligned Bounding Boxes (AABBs).
 *
 * This class constructs a binary bounding volume hierarchy tree for spatial partitioning
 * of centroid data. The tree accelerates spatial queries by hierarchically grouping
 * data points into nested bounding boxes.
 */
class BTree {
 public:
  /**
   * @brief Axis-Aligned Bounding Box (AABB) structure for spatial bounds representation.
   *
   * Represents a multidimensional bounding box defined by minimum and maximum coordinates
   * along each dimension. Used for spatial partitioning and collision detection.
   */
  struct AABB {
    std::vector<float> min;  ///< Minimum coordinates per dimension (inclusive bound)
    std::vector<float> max;  ///< Maximum coordinates per dimension (inclusive bound)

    /**
     * @brief Constructs an AABB with specified bounds.
     * @param min Minimum coordinates (default empty)
     * @param max Maximum coordinates (default empty)
     */
    AABB(const std::vector<float>& min = {}, const std::vector<float>& max = {});

    /**
     * @brief Identifies the axis with the largest spatial extent.
     * @return Index of the dimension (axis) with largest max-min difference
     * @pre AABB must be properly initialized with non-empty min/max vectors
     */
    int largestAxis() const;

    /**
     * @brief Calculates the largest dimension (size) of the AABB.
     * @return The maximum value of (max[i] - min[i]) across all dimensions
     */
    float largestDim() const;

    /**
     * @brief Computes AABB that bounds a subset of centroids.
     * @param centroids Source data table containing centroid coordinates
     * @param indices Indices of centroids to include in the bounding computation
     * @return Reference to this AABB after computation
     * @post AABB.min and AABB.max will encompass all specified centroids
     */
    AABB& fromCentroids(const DataTable* centroids, absl::Span<const uint32_t> indices);
  };

  /**
   * @brief Node structure for the binary bounding volume hierarchy tree.
   */
  struct BTreeNode {
    size_t count;                      ///< Number of centroid indices contained in this node and its descendants
    AABB aabb;                         ///< Bounding box enclosing all centroids in this subtree
    std::vector<uint32_t> indices;     ///< Centroid indices stored at this leaf node (empty for internal nodes)
    std::unique_ptr<BTreeNode> left;   ///< Left child subtree (nullptr for leaf nodes)
    std::unique_ptr<BTreeNode> right;  ///< Right child subtree (nullptr for leaf nodes)
  };

 public:
  DataTable* centroids;             ///< Pointer to the source centroid data table
  std::unique_ptr<BTreeNode> root;  ///< Root node of the BVH tree

  /**
   * @brief Constructs a BVH tree from centroid data.
   * @param centroids Pointer to the data table containing centroid coordinates
   * @post The tree is fully constructed and ready for spatial queries
   */
  BTree(DataTable* centroids);

 private:
  /**
   * @brief Recursively builds the BVH tree by spatial partitioning.
   *
   * Partitions the provided indices along the largest AABB axis and recursively
   * constructs left and right subtrees. Creates leaf nodes when the partition
   * size falls below a threshold.
   *
   * @param indices Span of centroid indices to partition in this recursion step
   * @return Unique pointer to the constructed subtree root
   * @note This method determines the tree's balance and spatial partitioning quality
   */
  std::unique_ptr<BTreeNode> recurse(absl::Span<uint32_t> indices);
};

}  // namespace splat
