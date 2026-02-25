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

#include <Eigen/Dense>
#include <cstdint>
#include <memory>
#include <vector>

namespace splat {

class DataTable;

class GaussianBVH {
 public:
  struct BVHBounds {
    Eigen::Vector3f min;
    Eigen::Vector3f max;
  };

  struct BVHNode {
    size_t count;
    BVHBounds bounds;
    std::vector<uint32_t> indices;
    std::unique_ptr<BVHNode> left;
    std::unique_ptr<BVHNode> right;

    BVHNode() = default;
    BVHNode(size_t count, const BVHBounds& bounds, std::vector<uint32_t> indices)
        : count(count), bounds(bounds), indices(std::move(indices)), left(nullptr), right(nullptr) {}
  };

 public:
  GaussianBVH(const DataTable* dataTable, const DataTable* extents);
  std::vector<uint32_t> queryOverlapping(const Eigen::Vector3f& boxMin, const Eigen::Vector3f& boxMax);

  size_t count() const { return root_ ? root_->count : 0; }
  BVHBounds sceneBounds() const { return root_ ? root_->bounds : BVHBounds(); }
  BVHNode* root() const { return root_.get(); }

 private:
  BVHBounds computeBound(absl::Span<uint32_t> indices);
  std::unique_ptr<BVHNode> buildNode(absl::Span<uint32_t> indices);
  void queryNode(const BVHNode* node, float minX, float minY, float minZ, float maxX, float maxY, float maxZ,
                 std::vector<uint32_t>& result);

 private:
  std::unique_ptr<BVHNode> root_;
  absl::Span<const float> x_;
  absl::Span<const float> y_;
  absl::Span<const float> z_;

  absl::Span<const float> extentX_;
  absl::Span<const float> extentY_;
  absl::Span<const float> extentZ_;

  static constexpr size_t MAX_LEAF_SIZE = 64u;
};

}  // namespace splat
