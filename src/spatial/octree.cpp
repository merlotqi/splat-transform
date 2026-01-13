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
#include <splat/spatial/octree.h>

#include <numeric>
#include <stdexcept>

namespace splat {

Octree::AABB::AABB(float minX, float minY, float minZ, float maxX, float maxY, float maxZ) {
  min = {minX, minY, minZ};
  max = {maxX, maxY, maxZ};
}

void Octree::AABB::getCenter(float& x, float& y, float& z) const {
  x = (min[0] + max[0]) * 0.5f;
  y = (min[1] + max[1]) * 0.5f;
  z = (min[2] + max[2]) * 0.5f;
}

bool Octree::AABB::contains(float x, float y, float z) const {
  return x >= min[0] && x < max[0] && y >= min[1] && y < max[1] && z >= min[2] && z < max[2];
}

Octree::Octree(DataTable* table, size_t maxPoints, int maxDepth)
    : dataTable_(table), maxPointPerNodes_(maxPoints), maxDepth_(maxDepth) {
  if (!dataTable_ || dataTable_->getNumRows() == 0) {
    throw std::invalid_argument("Input dataTable is invalid.");
  }

  const size_t numRows = dataTable_->getNumRows();
  const auto& colX = dataTable_->getColumnByName("x");
  const auto& colY = dataTable_->getColumnByName("y");
  const auto& colZ = dataTable_->getColumnByName("z");

  std::vector<size_t> allIndices(numRows);
  std::iota(allIndices.begin(), allIndices.end(), 0);

  float minX = std::numeric_limits<float>::max(), maxX = std::numeric_limits<float>::lowest();
  float minY = minX, maxY = maxX, minZ = minX, maxZ = maxX;

  for (size_t i = 0; i < numRows; ++i) {
    float x = colX.getValue(i), y = colY.getValue(i), z = colZ.getValue(i);
    minX = std::min(minX, x);
    maxX = std::max(maxX, x);
    minY = std::min(minY, y);
    maxY = std::max(maxY, y);
    minZ = std::min(minZ, z);
    maxZ = std::max(maxZ, z);
  }

  float eps = 1e-4f;
  AABB rootAABB(minX - eps, minY - eps, minZ - eps, maxX + eps, maxY + eps, maxZ + eps);

  this->root = build(rootAABB, absl::MakeSpan(allIndices), 0);
}

std::unique_ptr<Octree::OctreeNode> Octree::build(const AABB& aabb, absl::Span<size_t> indices, int depth) {
  auto node = std::make_unique<OctreeNode>();
  node->aabb = aabb;
  node->depth = depth;

  if (indices.size() <= static_cast<size_t>(maxPointPerNodes_) || depth >= maxDepth_) {
    node->isLeaf = true;
    node->pointIndices.assign(indices.begin(), indices.end());
    return node;
  }

  node->isLeaf = false;
  float cx, cy, cz;
  aabb.getCenter(cx, cy, cz);

  const auto& colX = dataTable_->getColumnByName("x");
  const auto& colY = dataTable_->getColumnByName("y");
  const auto& colZ = dataTable_->getColumnByName("z");

  auto splitZ = std::partition(indices.begin(), indices.end(), [&](size_t idx) { return colZ.getValue(idx) < cz; });
  absl::Span<size_t> zLow = indices.subspan(0, std::distance(indices.begin(), splitZ));
  absl::Span<size_t> zHigh = indices.subspan(zLow.size());

  auto partitionXY = [&](absl::Span<size_t> span, float centerX, float centerY) {
    auto splitY = std::partition(span.begin(), span.end(), [&](size_t idx) { return colY.getValue(idx) < centerY; });
    auto yLow = span.subspan(0, std::distance(span.begin(), splitY));
    auto yHigh = span.subspan(yLow.size());

    auto splitXLow = std::partition(yLow.begin(), yLow.end(), [&](size_t idx) { return colX.getValue(idx) < centerX; });
    auto splitXHigh =
        std::partition(yHigh.begin(), yHigh.end(), [&](size_t idx) { return colX.getValue(idx) < centerX; });

    return std::make_tuple(yLow.subspan(0, std::distance(yLow.begin(), splitXLow)),     // xLow, yLow
                           yLow.subspan(std::distance(yLow.begin(), splitXLow)),        // xHigh, yLow
                           yHigh.subspan(0, std::distance(yHigh.begin(), splitXHigh)),  // xLow, yHigh
                           yHigh.subspan(std::distance(yHigh.begin(), splitXHigh))      // xHigh, yHigh
    );
  };

  auto [q0, q1, q2, q3] = partitionXY(zLow, cx, cy);   // Z < cz
  auto [q4, q5, q6, q7] = partitionXY(zHigh, cx, cy);  // Z >= cz

  absl::Span<size_t> childrenSpans[8] = {q0, q1, q2, q3, q4, q5, q6, q7};

  for (int i = 0; i < 8; ++i) {
    if (childrenSpans[i].empty()) continue;

    AABB subAABB;
    subAABB.min[0] = (i & 1) ? cx : aabb.min[0];
    subAABB.max[0] = (i & 1) ? aabb.max[0] : cx;
    subAABB.min[1] = (i & 2) ? cy : aabb.min[1];
    subAABB.max[1] = (i & 2) ? aabb.max[1] : cy;
    subAABB.min[2] = (i & 4) ? cz : aabb.min[2];
    subAABB.max[2] = (i & 4) ? aabb.max[2] : cz;

    node->children[i] = build(subAABB, childrenSpans[i], depth + 1);
  }

  return node;
}

}  // namespace splat
