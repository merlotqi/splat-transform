/**
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
 */

#include <cuda_runtime.h>
#include <splat/gpu/gpu-clustering.h>

#include <cfloat>

namespace splat {

__global__ void cluster_kernel(const float* __restrict__ points, const float* __restrict__ centroids,
                               unsigned int* __restrict__ results, int numPoints, int numCentroids, int numColumns) {
  extern __shared__ float sharedCentroids[];

  int pointIdx = blockIdx.x * blockDim.x + threadIdx.x;
  int localIdx = threadIdx.x;
  const int chunkSize = blockDim.x;

  float minDistanceSqr = FLT_MAX;
  unsigned int bestCentroid = 0;

  float currentPoint[64];
  if (pointIdx < numPoints) {
    for (int i = 0; i < numColumns; i++) {
      currentPoint[i] = points[pointIdx * numColumns + i];
    }
  }

  int numChunks = (numCentroids + chunkSize - 1) / chunkSize;
  for (int i = 0; i < numChunks; i++) {
    int centroidToLoad = i * chunkSize + localIdx;
    if (centroidToLoad < numCentroids) {
      for (int d = 0; d < numColumns; d++) {
        sharedCentroids[localIdx * numColumns + d] = centroids[centroidToLoad * numColumns + d];
      }
    }
    __syncthreads();

    if (pointIdx < numPoints) {
      int currentChunkSize = min(chunkSize, numCentroids - i * chunkSize);
      for (int c = 0; c < currentChunkSize; c++) {
        float distSqr = 0.0f;
        for (int d = 0; d < numColumns; d++) {
          float diff = currentPoint[d] - sharedCentroids[c * numColumns + d];
          distSqr += diff * diff;
        }
        if (distSqr < minDistanceSqr) {
          minDistanceSqr = distSqr;
          bestCentroid = i * chunkSize + c;
        }
      }
    }
    __syncthreads();
  }

  if (pointIdx < numPoints) {
    results[pointIdx] = bestCentroid;
  }
}

void gpu_cluster_3d_execute(DataTable* points, const DataTable* centroids, std::vector<uint32_t>& labels) {
  if (!points || !centroids) return;

  const int numPoints = static_cast<int>(points->getNumRows());
  const int numCentroids = static_cast<int>(centroids->getNumRows());
  const int numColumns = static_cast<int>(points->getNumColumns());

  auto interleave = [&](const splat::DataTable* table, int startRow, int rowCount) {
    std::vector<float> buffer(rowCount * numColumns);
    for (int c = 0; c < numColumns; ++c) {
      const auto& column = table->getColumn(c);
      for (int r = 0; r < rowCount; ++r) {
        buffer[r * numColumns + c] = column.getValue<float>(startRow + r);
      }
    }
    return buffer;
  };

  std::vector<float> h_centroids_interleaved = interleave(centroids, 0, numCentroids);

  float *d_points, *d_centroids;
  unsigned int* d_results;
  const int batchSize = 65536;

  cudaMalloc(&d_centroids, h_centroids_interleaved.size() * sizeof(float));
  cudaMalloc(&d_points, batchSize * numColumns * sizeof(float));
  cudaMalloc(&d_results, batchSize * sizeof(unsigned int));

  cudaMemcpy(d_centroids, h_centroids_interleaved.data(), h_centroids_interleaved.size() * sizeof(float),
             cudaMemcpyHostToDevice);

  for (int offset = 0; offset < numPoints; offset += batchSize) {
    int currentBatchSize = std::min(batchSize, numPoints - offset);

    std::vector<float> h_points_batch = interleave(points, offset, currentBatchSize);

    cudaMemcpy(d_points, h_points_batch.data(), h_points_batch.size() * sizeof(float), cudaMemcpyHostToDevice);

    int threadsPerBlock = 128;
    int blocksPerGrid = (currentBatchSize + threadsPerBlock - 1) / threadsPerBlock;
    size_t sharedMemSize = threadsPerBlock * numColumns * sizeof(float);

    cluster_kernel<<<blocksPerGrid, threadsPerBlock, sharedMemSize>>>(d_points, d_centroids, d_results,
                                                                      currentBatchSize, numCentroids, numColumns);

    cudaMemcpy(labels.data() + offset, d_results, currentBatchSize * sizeof(unsigned int), cudaMemcpyDeviceToHost);
  }

  cudaFree(d_points);
  cudaFree(d_centroids);
  cudaFree(d_results);
}

}  // namespace splat
