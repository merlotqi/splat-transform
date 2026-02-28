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

#include <cuda_runtime.h>
#include <splat/models/data-table.h>

#include <cmath>
#include <vector>

namespace splat {

constexpr int TILE_SIZE = 64;
constexpr int VOXELS_PER_BLOCK = 64;
constexpr int MAX_BLOCKS_PER_BATCH = 4096;

struct BatchInfoGPU {
  uint32_t indexOffset;
  uint32_t indexCount;
  uint32_t numBlocksX;
  uint32_t numBlocksY;
  uint32_t numBlocksZ;
  float blockMinX;
  float blockMinY;
  float blockMinZ;
};

struct GaussianSoADevice {
  const float* x;
  const float* y;
  const float* z;
  const float* opacity;
  const float* rotW;
  const float* rotX;
  const float* rotY;
  const float* rotZ;
  const float* scaleX;
  const float* scaleY;
  const float* scaleZ;
  const float* extentX;
  const float* extentY;
  const float* extentZ;
};

__device__ inline float3 cross3(const float3& a, const float3& b) {
  return make_float3(a.y * b.z - a.z * b.y, a.z * b.x - a.x * b.z, a.x * b.y - a.y * b.x);
}

__device__ inline float dot3(const float3& a, const float3& b) { return a.x * b.x + a.y * b.y + a.z * b.z; }

__device__ inline float3 operator+(const float3& a, const float3& b) {
  return make_float3(a.x + b.x, a.y + b.y, a.z + b.z);
}

__device__ inline float3 operator-(const float3& a, const float3& b) {
  return make_float3(a.x - b.x, a.y - b.y, a.z - b.z);
}

__device__ inline float3 operator*(const float3& a, float b) { return make_float3(a.x * b, a.y * b, a.z * b); }

__device__ float evaluateGaussian(int gi, const GaussianSoADevice& g, const float3& voxelCenter, float voxelHalfSize) {
  float3 center = make_float3(g.x[gi], g.y[gi], g.z[gi]);

  float3 diff = voxelCenter - center;

  float3 extent = make_float3(g.extentX[gi], g.extentY[gi], g.extentZ[gi]);

  if (fabsf(diff.x) > extent.x + voxelHalfSize || fabsf(diff.y) > extent.y + voxelHalfSize ||
      fabsf(diff.z) > extent.z + voxelHalfSize)
    return 0.0f;

  float3 minCorner = voxelCenter - make_float3(voxelHalfSize, voxelHalfSize, voxelHalfSize);
  float3 maxCorner = voxelCenter + make_float3(voxelHalfSize, voxelHalfSize, voxelHalfSize);

  float3 closest =
      make_float3(fminf(fmaxf(center.x, minCorner.x), maxCorner.x), fminf(fmaxf(center.y, minCorner.y), maxCorner.y),
                  fminf(fmaxf(center.z, minCorner.z), maxCorner.z));

  float3 closestDiff = closest - center;

  float3 qxyz = make_float3(-g.rotX[gi], -g.rotY[gi], -g.rotZ[gi]);

  float3 t = cross3(qxyz, closestDiff) * 2.0f;

  float3 local = closestDiff + t * g.rotW[gi] + cross3(qxyz, t);

  float3 invScale = make_float3(expf(-g.scaleX[gi]), expf(-g.scaleY[gi]), expf(-g.scaleZ[gi]));

  float3 scaled = make_float3(local.x * invScale.x, local.y * invScale.y, local.z * invScale.z);

  float d2 = dot3(scaled, scaled);

  float opacity = 1.0f / (1.0f + expf(-g.opacity[gi]));

  return opacity * expf(-0.5f * d2);
}

__device__ inline void mortonToXYZ(uint32_t m, uint32_t& x, uint32_t& y, uint32_t& z) {
  x = (m & 1u) | ((m >> 2u) & 2u);
  y = ((m >> 1u) & 1u) | ((m >> 3u) & 2u);
  z = ((m >> 2u) & 1u) | ((m >> 4u) & 2u);
}

__global__ void voxelizeKernel(GaussianSoADevice g, const uint32_t* indices, const BatchInfoGPU* batchInfos,
                               uint32_t* results, float voxelResolution, float opacityCutoff,
                               uint32_t maxBlocksPerBatch) {
  __shared__ uint32_t sharedIndices[TILE_SIZE];

  uint32_t voxelIdx = threadIdx.x;
  uint32_t flatBlockId = blockIdx.x;
  uint32_t batchIdx = blockIdx.y;

  const BatchInfoGPU& info = batchInfos[batchIdx];

  uint32_t totalBlocks = info.numBlocksX * info.numBlocksY * info.numBlocksZ;

  if (flatBlockId >= totalBlocks) return;

  uint32_t blockX = flatBlockId % info.numBlocksX;
  uint32_t blockY = (flatBlockId / info.numBlocksX) % info.numBlocksY;
  uint32_t blockZ = flatBlockId / (info.numBlocksX * info.numBlocksY);

  uint32_t lx, ly, lz;
  mortonToXYZ(voxelIdx, lx, ly, lz);

  float3 blockMin = make_float3(info.blockMinX, info.blockMinY, info.blockMinZ);

  float3 blockOffset =
      make_float3(blockX * 4.0f * voxelResolution, blockY * 4.0f * voxelResolution, blockZ * 4.0f * voxelResolution);

  float3 voxelCenter =
      blockMin + blockOffset +
      make_float3((lx + 0.5f) * voxelResolution, (ly + 0.5f) * voxelResolution, (lz + 0.5f) * voxelResolution);

  float voxelHalf = voxelResolution * 0.5f;

  float totalSigma = 0.0f;

  uint32_t numTiles = (info.indexCount + TILE_SIZE - 1) / TILE_SIZE;

  for (uint32_t tile = 0; tile < numTiles; tile++) {
    uint32_t loadIdx = tile * TILE_SIZE + voxelIdx;

    if (loadIdx < info.indexCount) sharedIndices[voxelIdx] = indices[info.indexOffset + loadIdx];

    __syncthreads();

    uint32_t thisTile = min((uint32_t)TILE_SIZE, info.indexCount - tile * TILE_SIZE);

    if (totalSigma < 7.0f) {
      for (uint32_t c = 0; c < thisTile; c++) {
        totalSigma += evaluateGaussian(sharedIndices[c], g, voxelCenter, voxelHalf);

        if (totalSigma >= 7.0f) break;
      }
    }

    __syncthreads();
  }

  float finalOpacity = 1.0f - expf(-totalSigma);
  bool isSolid = finalOpacity >= opacityCutoff;

  if (isSolid) {
    uint32_t linearIdx = lz * 16u + ly * 4u + lx;

    uint32_t resultBase = batchIdx * maxBlocksPerBatch * 2u;

    uint32_t wordIndex = resultBase + flatBlockId * 2u + (linearIdx >> 5u);

    uint32_t bitIndex = linearIdx & 31u;

    atomicOr(&results[wordIndex], 1u << bitIndex);
  }
}

class CudaVoxelizer {
 public:
  void voxelize(const DataTable& table, const DataTable& extents, const std::vector<uint32_t>& indices,
                const std::vector<BatchInfoGPU>& batches, float voxelResolution, float opacityCutoff,
                std::vector<uint32_t>& outMasks) {
    size_t numGaussians = table.getNumRows();

    auto& x = table.getColumnByName("x").asVector<float>();
    auto& y = table.getColumnByName("y").asVector<float>();
    auto& z = table.getColumnByName("z").asVector<float>();
    auto& opacity = table.getColumnByName("opacity").asVector<float>();
    auto& rotW = table.getColumnByName("rot_0").asVector<float>();
    auto& rotX = table.getColumnByName("rot_1").asVector<float>();
    auto& rotY = table.getColumnByName("rot_2").asVector<float>();
    auto& rotZ = table.getColumnByName("rot_3").asVector<float>();
    auto& scaleX = table.getColumnByName("scale_0").asVector<float>();
    auto& scaleY = table.getColumnByName("scale_1").asVector<float>();
    auto& scaleZ = table.getColumnByName("scale_2").asVector<float>();
    auto& extentX = extents.getColumnByName("extent_x").asVector<float>();
    auto& extentY = extents.getColumnByName("extent_y").asVector<float>();
    auto& extentZ = extents.getColumnByName("extent_z").asVector<float>();

    float *dx, *dy, *dz, *dopacity;
    float *drotW, *drotX, *drotY, *drotZ;
    float *dscaleX, *dscaleY, *dscaleZ;
    float *dextentX, *dextentY, *dextentZ;

#define CUDA_ALLOC_COPY(ptr, vec)               \
  cudaMalloc(&ptr, vec.size() * sizeof(float)); \
  cudaMemcpy(ptr, vec.data(), vec.size() * sizeof(float), cudaMemcpyHostToDevice);

    CUDA_ALLOC_COPY(dx, x);
    CUDA_ALLOC_COPY(dy, y);
    CUDA_ALLOC_COPY(dz, z);
    CUDA_ALLOC_COPY(dopacity, opacity);
    CUDA_ALLOC_COPY(drotW, rotW);
    CUDA_ALLOC_COPY(drotX, rotX);
    CUDA_ALLOC_COPY(drotY, rotY);
    CUDA_ALLOC_COPY(drotZ, rotZ);
    CUDA_ALLOC_COPY(dscaleX, scaleX);
    CUDA_ALLOC_COPY(dscaleY, scaleY);
    CUDA_ALLOC_COPY(dscaleZ, scaleZ);
    CUDA_ALLOC_COPY(dextentX, extentX);
    CUDA_ALLOC_COPY(dextentY, extentY);
    CUDA_ALLOC_COPY(dextentZ, extentZ);

    GaussianSoADevice g{dx,    dy,      dz,      dopacity, drotW,    drotX,    drotY,
                        drotZ, dscaleX, dscaleY, dscaleZ,  dextentX, dextentY, dextentZ};

    // indices
    uint32_t* dIndices;
    cudaMalloc(&dIndices, indices.size() * sizeof(uint32_t));
    cudaMemcpy(dIndices, indices.data(), indices.size() * sizeof(uint32_t), cudaMemcpyHostToDevice);

    // batch info
    BatchInfoGPU* dBatch;
    cudaMalloc(&dBatch, batches.size() * sizeof(BatchInfoGPU));
    cudaMemcpy(dBatch, batches.data(), batches.size() * sizeof(BatchInfoGPU), cudaMemcpyHostToDevice);

    // result
    size_t resultCount = batches.size() * MAX_BLOCKS_PER_BATCH * 2;

    uint32_t* dResults;
    cudaMalloc(&dResults, resultCount * sizeof(uint32_t));
    cudaMemset(dResults, 0, resultCount * sizeof(uint32_t));

    // launch
    dim3 grid(MAX_BLOCKS_PER_BATCH, batches.size());

    dim3 block(VOXELS_PER_BLOCK);

    voxelizeKernel<<<grid, block>>>(g, dIndices, dBatch, dResults, voxelResolution, opacityCutoff,
                                    MAX_BLOCKS_PER_BATCH);

    cudaDeviceSynchronize();

    outMasks.resize(resultCount);
    cudaMemcpy(outMasks.data(), dResults, resultCount * sizeof(uint32_t), cudaMemcpyDeviceToHost);

    // free
    cudaFree(dIndices);
    cudaFree(dBatch);
    cudaFree(dResults);

    cudaFree(dx);
    cudaFree(dy);
    cudaFree(dz);
    cudaFree(dopacity);
    cudaFree(drotW);
    cudaFree(drotX);
    cudaFree(drotY);
    cudaFree(drotZ);
    cudaFree(dscaleX);
    cudaFree(dscaleY);
    cudaFree(dscaleZ);
    cudaFree(dextentX);
    cudaFree(dextentY);
    cudaFree(dextentZ);
  }
};

}  // namespace splat
