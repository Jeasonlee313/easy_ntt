//
// Created by jeason on 1/6/26.
//

#include <cstdio>
#include <iostream>

#include "driver/cuda_macros.h"
#include "kernels/voxelization.cuh"

namespace kernels {

/**
 * @brief reorder the point cloud and fill it into the voxels.
 * @param max_num_points the max number of points per batch
 * @param points points data ptr, [sum(points size * point value num)]
 * @param points_size the actually point number per batch, ptr [batch size]
 * @param min/max_*_range pc range
 * @param pillar_*_size per pillar size
 * @param grid_y_size the pillar num at y axis
 * @param grid_x_size the pillar num at x axis
 * @param num_point_values point dim, usually is 4 for XYZI
 * @param max_points_per_voxel
 * @param mask @output the points num in this pillar, ptr [batch size * grid_y_size * grid_x_size]
 * @param voxels @output the voxels src data, ptr [batch size * grid_y_size * grid_x_size *
 * max_points_per_voxel * num_point_value]
 */
__global__ void generateVoxelsKernel(const int max_num_points, const float *const __restrict__ points,
                                     const unsigned int *const __restrict__ points_size, const float min_x_range,
                                     const float max_x_range, const float min_y_range, const float max_y_range,
                                     const float min_z_range, const float max_z_range, const float pillar_x_size,
                                     const float pillar_y_size, const float pillar_z_size, const int grid_y_size,
                                     const int grid_x_size, const int num_point_values, const int max_points_per_voxel,
                                     unsigned int *mask, float *voxels) {
  int point_idx = blockIdx.x * blockDim.x + threadIdx.x;
  int batch_idx = point_idx / max_num_points;
  int point_idx_in_frame = point_idx % max_num_points;
  if (point_idx_in_frame >= points_size[batch_idx]) {
    return;
  }
  float px = points[num_point_values * point_idx];
  float py = points[num_point_values * point_idx + 1];
  float pz = points[num_point_values * point_idx + 2];
  float pw = points[num_point_values * point_idx + 3];
  float pt;
  if (num_point_values == 5) {
    pt = points[num_point_values * point_idx + 4];
  }
  if (px < min_x_range || px >= max_x_range || py < min_y_range || py >= max_y_range || pz < min_z_range ||
      pz >= max_z_range) {
    return;
  }
  int voxel_idx = floorf((px - min_x_range) / pillar_x_size);
  int voxel_idy = floorf((py - min_y_range) / pillar_y_size);
  unsigned int voxel_index = (batch_idx * grid_y_size + voxel_idy) * grid_x_size + voxel_idx;
  unsigned int point_id = atomicAdd(&(mask[voxel_index]), 1);
  if (point_id >= max_points_per_voxel) {
    return;
  }
  float *address = voxels + (voxel_index * max_points_per_voxel + point_id) * num_point_values;
  atomicExch(address + 0, px);
  atomicExch(address + 1, py);
  atomicExch(address + 2, pz);
  atomicExch(address + 3, pw);
  if (num_point_values == 5) {
    atomicExch(address + 4, pt);
  }
}

/**
 * @brief generate base voxel feature and coords, base feature is xyzi, coords is 0, 0, vy_id, vx_id
 * @param batch_size batch size of point cloud
 * @param mask voxel count in last kernel, ptr [batch size * grid_y_size * grid_x_size]
 * @param voxels the voxels src data, ptr [batch size * grid_y_size * grid_x_size *
 * max_points_per_voxel * num_point_value]
 * @param grid_y_size the pillar num at y axis
 * @param grid_x_size the pillar num at x axis
 * @param max_pillar_num max pillar num per batch
 * @param max_points_per_voxel
 * @param num_point_values value num per point
 * @param pillar_num the pillar num in per batch, ptr [batch size]
 * @param voxel_features the voxel
 * @param voxel_num_points the voxel squeeze feature, ptr [batch size * grid_x_size * grid_y_size *
 * num_point_values]
 * @param coords the voxel coords, ptr [batch size * max_pillar_num * 4]
 */
__global__ void generateBaseFeaturesKernel(const int batch_size, const unsigned int *__restrict__ const mask,
                                           const float *__restrict__ const voxels, const int grid_y_size,
                                           const int grid_x_size, const int max_pillar_num,
                                           const int max_points_per_voxel, const int num_point_values,
                                           unsigned int *pillar_num, float *voxel_features,
                                           unsigned int *voxel_num_points, unsigned int *coords) {
  int voxel_id = blockIdx.x * blockDim.x + threadIdx.x;
  int voxel_idx = voxel_id % grid_x_size;
  int voxel_idy = (voxel_id / grid_x_size) % grid_y_size;
  int batch_id = voxel_id / (grid_y_size * grid_x_size);
  if (batch_id >= batch_size) {
    return;
  }
  unsigned int count = mask[voxel_id];
  if (!(count > 0)) {
    return;
  }
  count = count < max_points_per_voxel ? count : max_points_per_voxel;
  int current_pillar_id = 0;
  current_pillar_id = atomicAdd(pillar_num + batch_id, 1);
  voxel_num_points[batch_id * max_pillar_num + current_pillar_id] = count;
  int4 coord = {0, 0, voxel_idy, voxel_idx};
  ((int4 *)coords)[batch_id * max_pillar_num + current_pillar_id] = coord;

  if (num_point_values == 4) {
#pragma unroll
    for (int i = 0; i < count; ++i) {
      int in_index = voxel_id * max_points_per_voxel + i;
      int out_index = (batch_id * max_pillar_num + current_pillar_id) * max_points_per_voxel + i;
      ((float4 *)voxel_features)[out_index] = ((float4 *)voxels)[in_index];
    }
  } else {
#pragma unroll
    for (int i = 0; i < count; ++i) {
      int in_index = voxel_id * max_points_per_voxel + i;
      int out_index = (batch_id * max_pillar_num + current_pillar_id) * max_points_per_voxel + i;
#pragma unroll 5
      for (int k = 0; k < 5; ++k) {
        voxel_features[5 * out_index + k] = voxels[5 * in_index + k];
      }
    }
  }
}

void generateVoxels(const int batch_size, const int max_num_points, const float *points,
                    const unsigned int *points_size, const float min_x_range, const float max_x_range,
                    const float min_y_range, const float max_y_range, const float min_z_range, const float max_z_range,
                    const float pillar_x_size, const float pillar_y_size, const float pillar_z_size,
                    const int grid_y_size, const int grid_x_size, const int num_point_values,
                    const int max_points_per_voxel, unsigned int *mask, float *voxels, cudaStream_t stream) {
  int thread_num = 256;
  dim3 blocks((batch_size * max_num_points + thread_num - 1) / thread_num);
  dim3 threads(thread_num);
  generateVoxelsKernel<<<blocks, threads, 0, stream>>>(
      max_num_points, points, points_size, min_x_range, max_x_range, min_y_range, max_y_range, min_z_range, max_z_range,
      pillar_x_size, pillar_y_size, pillar_z_size, grid_y_size, grid_x_size, num_point_values, max_points_per_voxel,
      mask, voxels);
}

void generateBaseFeatures(const int batch_size, const unsigned int *mask, const float *voxels, const int grid_y_size,
                          const int grid_x_size, const int max_pillar_num, const int max_points_per_voxel,
                          const int num_point_values, unsigned int *pillar_num, float *voxel_features,
                          unsigned int *voxel_num_points, unsigned int *coords, cudaStream_t stream) {
  int block_size = 1024;
  dim3 threads(block_size);
  dim3 blocks((batch_size * grid_y_size * grid_x_size + block_size - 1) / block_size);
  generateBaseFeaturesKernel<<<blocks, threads, 0, stream>>>(batch_size, mask, voxels, grid_y_size, grid_x_size,
                                                             max_pillar_num, max_points_per_voxel, num_point_values,
                                                             pillar_num, voxel_features, voxel_num_points, coords);
}

__global__ void generateFeaturesKernel4x(const int batch_size, const float *__restrict__ const voxel_features,
                                         const unsigned int *__restrict__ const coords, const float voxel_x,
                                         const float voxel_y, const float voxel_z, const float range_min_x,
                                         const float range_min_y, const float range_min_z,
                                         const unsigned int voxel_features_size, const unsigned int max_points,
                                         const unsigned int max_voxels, unsigned int *voxel_num_points,
                                         unsigned int *pillar_num, float *features) {
  int warp_size = max_points;
  int pillar_idx = blockIdx.x * 4 + threadIdx.x / warp_size;
  int point_idx = threadIdx.x % warp_size;
  // In case the actual number of points is less than warp_size
  // E.g., warp_size=32, max_points=20
  if (point_idx >= max_points) {
    return;
  }
  int batch_idx = pillar_idx / max_voxels;
  if (batch_idx >= batch_size) {
    return;
  }
  int pillar_idx_in_frame = pillar_idx % max_voxels;
  int pillar_idx_in_block = threadIdx.x / warp_size;
  // Limit number of voxels to max_voxels
  unsigned int num_pillars = pillar_num[batch_idx] > max_voxels ? max_voxels : pillar_num[batch_idx];
  // Update max_voxel to actual number
  if (pillar_idx_in_frame == 0 && point_idx == 0) {
    pillar_num[batch_idx] = num_pillars;
  }
  if (pillar_idx_in_frame >= num_pillars) {
    return;
  }
  // load src
  __shared__ float4 pillar_share_mem[4][64];         // up to 64 points per pillar
  __shared__ float4 pillar_sum_share_mem[4];         // 4*4
  __shared__ int4 coords_share_mem[4];               // 4*4
  __shared__ int points_num_share_mem[4];            // 4
  __shared__ float pillar_out_share_mem[4][64][10];  // up to 11 output features per point

  if (point_idx == 0) {
    points_num_share_mem[pillar_idx_in_block] = voxel_num_points[pillar_idx];
    coords_share_mem[pillar_idx_in_block] = ((int4 *)coords)[pillar_idx];
    pillar_sum_share_mem[pillar_idx_in_block] = {0, 0, 0, 0};
  }
  pillar_share_mem[pillar_idx_in_block][point_idx] = ((float4 *)voxel_features)[pillar_idx * max_points + point_idx];
  __syncthreads();
  // calculate sm
  if (point_idx < points_num_share_mem[pillar_idx_in_block]) {
    atomicAdd(&(pillar_sum_share_mem[pillar_idx_in_block].x), pillar_share_mem[pillar_idx_in_block][point_idx].x);
    atomicAdd(&(pillar_sum_share_mem[pillar_idx_in_block].y), pillar_share_mem[pillar_idx_in_block][point_idx].y);
    atomicAdd(&(pillar_sum_share_mem[pillar_idx_in_block].z), pillar_share_mem[pillar_idx_in_block][point_idx].z);
  }
  __syncthreads();
  // feature-mean
  float4 mean;
  float valid_points = points_num_share_mem[pillar_idx_in_block];
  mean.x = pillar_sum_share_mem[pillar_idx_in_block].x / valid_points;
  mean.y = pillar_sum_share_mem[pillar_idx_in_block].y / valid_points;
  mean.z = pillar_sum_share_mem[pillar_idx_in_block].z / valid_points;
  mean.x = pillar_share_mem[pillar_idx_in_block][point_idx].x - mean.x;
  mean.y = pillar_share_mem[pillar_idx_in_block][point_idx].y - mean.y;
  mean.z = pillar_share_mem[pillar_idx_in_block][point_idx].z - mean.z;
  // calculate offset
  float x_offset = voxel_x / 2.0f + coords_share_mem[pillar_idx_in_block].w * voxel_x + range_min_x;
  float y_offset = voxel_y / 2.0f + coords_share_mem[pillar_idx_in_block].z * voxel_y + range_min_y;
  float z_offset = voxel_z / 2.0f + coords_share_mem[pillar_idx_in_block].y * voxel_z + range_min_z;
  // feature-offset
  float4 center;
  center.x = pillar_share_mem[pillar_idx_in_block][point_idx].x - x_offset;
  center.y = pillar_share_mem[pillar_idx_in_block][point_idx].y - y_offset;
  center.z = pillar_share_mem[pillar_idx_in_block][point_idx].z - z_offset;
  // store output
  if (point_idx < points_num_share_mem[pillar_idx_in_block]) {
    pillar_out_share_mem[pillar_idx_in_block][point_idx][0] = pillar_share_mem[pillar_idx_in_block][point_idx].x;
    pillar_out_share_mem[pillar_idx_in_block][point_idx][1] = pillar_share_mem[pillar_idx_in_block][point_idx].y;
    pillar_out_share_mem[pillar_idx_in_block][point_idx][2] = pillar_share_mem[pillar_idx_in_block][point_idx].z;
    pillar_out_share_mem[pillar_idx_in_block][point_idx][3] = pillar_share_mem[pillar_idx_in_block][point_idx].w;
    pillar_out_share_mem[pillar_idx_in_block][point_idx][4] = mean.x;
    pillar_out_share_mem[pillar_idx_in_block][point_idx][5] = mean.y;
    pillar_out_share_mem[pillar_idx_in_block][point_idx][6] = mean.z;
    pillar_out_share_mem[pillar_idx_in_block][point_idx][7] = center.x;
    pillar_out_share_mem[pillar_idx_in_block][point_idx][8] = center.y;
    pillar_out_share_mem[pillar_idx_in_block][point_idx][9] = center.z;

  } else {
    pillar_out_share_mem[pillar_idx_in_block][point_idx][0] = 0;
    pillar_out_share_mem[pillar_idx_in_block][point_idx][1] = 0;
    pillar_out_share_mem[pillar_idx_in_block][point_idx][2] = 0;
    pillar_out_share_mem[pillar_idx_in_block][point_idx][3] = 0;
    pillar_out_share_mem[pillar_idx_in_block][point_idx][4] = 0;
    pillar_out_share_mem[pillar_idx_in_block][point_idx][5] = 0;
    pillar_out_share_mem[pillar_idx_in_block][point_idx][6] = 0;
    pillar_out_share_mem[pillar_idx_in_block][point_idx][7] = 0;
    pillar_out_share_mem[pillar_idx_in_block][point_idx][8] = 0;
    pillar_out_share_mem[pillar_idx_in_block][point_idx][9] = 0;
  }
  __syncthreads();
  for (int i = 0; i < voxel_features_size; i++) {
    unsigned int output_share_mem_id = pillar_idx_in_block * 64 * 10 + point_idx * 10 + i;
    unsigned int output_id = pillar_idx * max_points * voxel_features_size + point_idx * voxel_features_size + i;
    features[output_id] = ((float *)pillar_out_share_mem)[output_share_mem_id];
  }
}

void generateFeatures(const int batch_size, const float *voxel_features, const unsigned int *coords,
                      const float voxel_x, const float voxel_y, const float voxel_z, const float range_min_x,
                      const float range_min_y, const float range_min_z, const unsigned int voxel_features_size,
                      const unsigned int max_points, const unsigned int max_voxels, const unsigned int num_point_values,
                      unsigned int *voxel_num_points, unsigned int *pillar_nums, float *features, cudaStream_t stream) {
  unsigned int warp_size = max_points;
  dim3 blocks((batch_size * max_voxels + 3) / 4);
  dim3 threads(4 * warp_size);
  if (num_point_values == 4) {
    generateFeaturesKernel4x<<<blocks, threads, 0, stream>>>(
        batch_size, voxel_features, coords, voxel_x, voxel_y, voxel_z, range_min_x, range_min_y, range_min_z,
        voxel_features_size, max_points, max_voxels, voxel_num_points, pillar_nums, features);
  } else {
    CUERROR << "num_point_values is not supported";
  }
}

void generateVoxelizationFeature(const int batch_size, const int max_num_points, const float *points,
                                 const unsigned int *point_size, const float min_x_range, const float max_x_range,
                                 const float min_y_range, const float max_y_range, const float min_z_range,
                                 const float max_z_range, const float pillar_x_size, const float pillar_y_size,
                                 const float pillar_z_size, const unsigned int num_point_value,
                                 const int max_points_per_voxel, const int max_pillar_num,
                                 const unsigned int num_pillar_feature_value, unsigned int *point_num_voxel,
                                 float *voxel_feature, unsigned int *pillar_num, float *squeeze_voxel_feature,
                                 unsigned int *squeeze_voxel_num_points, unsigned int *pillar_coords,
                                 float *pillar_feature, cudaStream_t stream) {
  int grid_y_size = static_cast<int>(ceil((max_y_range - min_y_range) / pillar_y_size));
  int grid_x_size = static_cast<int>(ceil((max_x_range - min_x_range) / pillar_x_size));
  generateVoxels(batch_size, max_num_points, points, point_size, min_x_range, max_x_range, min_y_range, max_y_range,
                 min_z_range, max_z_range, pillar_x_size, pillar_y_size, pillar_z_size, grid_y_size, grid_x_size,
                 num_point_value, max_points_per_voxel, point_num_voxel, voxel_feature, stream);

  generateBaseFeatures(batch_size, point_num_voxel, voxel_feature, grid_y_size, grid_x_size, max_pillar_num,
                       max_points_per_voxel, num_point_value, pillar_num, squeeze_voxel_feature,
                       squeeze_voxel_num_points, pillar_coords, stream);

  generateFeatures(batch_size, squeeze_voxel_feature, pillar_coords, pillar_x_size, pillar_y_size, pillar_z_size,
                   min_x_range, min_y_range, min_z_range, num_pillar_feature_value, max_points_per_voxel,
                   max_pillar_num, num_point_value, squeeze_voxel_num_points, pillar_num, pillar_feature, stream);
}
}  // namespace kernels
