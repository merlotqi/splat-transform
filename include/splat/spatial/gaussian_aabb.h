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

#include <splat/models/data-table.h>

#include <Eigen/Dense>

namespace splat {

struct GaussianExtentsResult {
  /**
   * DataTable containing extent_x, extent_y, extent_z columns.
   * To compute AABB for Gaussian i:
   *   minX = x[i] - extent_x[i], maxX = x[i] + extent_x[i]
   *   minY = y[i] - extent_y[i], maxY = y[i] + extent_y[i]
   *   minZ = z[i] - extent_z[i], maxZ = z[i] + extent_z[i]
   */
  std::unique_ptr<DataTable> extents{nullptr};

  /** Scene bounds (union of all Gaussian AABBs) */
  struct {
    Eigen::Vector3f min;
    Eigen::Vector3f max;
  } sceneBounds;
  /** Number of Gaussians skipped due to invalid values */
  size_t invalidCount{0};
};

/**
 * Compute axis-aligned bounding box half-extents for all Gaussians in a DataTable.
 *
 * Each Gaussian is an oriented ellipsoid defined by position, rotation (quaternion),
 * and scale (log scale). This function computes the AABB that encloses each
 * rotated ellipsoid and stores only the half-extents. The full AABB can be
 * reconstructed at runtime using: min = position - extent, max = position + extent.
 *
 * @param dataTable - DataTable containing Gaussian splat data
 * @returns GaussianExtentsResult with extents DataTable and scene bounds
 */
GaussianExtentsResult computeGaussianExtents(const DataTable* dataTable);

}  // namespace splat
