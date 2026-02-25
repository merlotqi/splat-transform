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

namespace splat {

/**
 * Voxelizes Gaussian splat data and writes the result as a sparse voxel octree.
 *
 * This function performs GPU-accelerated voxelization of Gaussian splat data
 * and outputs two files:
 * - `filename` (.voxel.json) - JSON metadata including bounds, resolution, and array sizes
 * - Corresponding .voxel.bin - Binary octree data (nodes + leafData as uint32_t arrays)
 *
 * The binary file layout is:
 * - Bytes 0 to (nodeCount * 4 - 1): nodes array (uint32_t, little-endian)
 * - Bytes (nodeCount * 4) to end: leafData array (uint32_t, little-endian)
 *
 * @param filename - Output filename (without extension) or with .voxel.json extension
 * @param dataTable - Gaussian splat data to voxelize
 * @param voxelResolution - Size of each voxel in world units. Default: 0.05
 * @param opacityCutoff - Opacity threshold for solid voxels - voxels below this are considered empty. Default: 0.5
 *
 * @return True if voxelization succeeded, false otherwise
 *
 * @note This function requires a GPU that supports compute shaders.
 *
 */
void writeVoxel(const std::string& filename, const DataTable* dataTable, float voxelResolution = 0.05f,
                float opacityCutoff = 0.5f);

}  // namespace splat
