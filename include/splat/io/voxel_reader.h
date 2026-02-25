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
 * Read a .voxel.json file and convert to DataTable (finest/leaf LOD).
 *
 * Loads the voxel octree from .voxel.json + .voxel.bin, traverses to the leaf level,
 * and outputs a DataTable in the same Gaussian splat format as voxel-octree-node.mjs
 * at the leaf level. Users can then save to PLY, CSV, or any other format.
 *
 * @param fileSystem - File system for reading files
 * @param filename - Path to .voxel.json (the .voxel.bin must exist alongside it)
 * @returns DataTable with voxel block centers as Gaussian splats
 */
std::unique_ptr<DataTable> readVoxel(const std::string& filename);

}  // namespace splat
