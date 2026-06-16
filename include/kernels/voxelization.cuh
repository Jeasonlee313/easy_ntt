//
// Created by jeason on 1/6/26.
//

#ifndef NV_INFRA_VOXELIZATION_CUH
#define NV_INFRA_VOXELIZATION_CUH
#include <cuda_runtime_api.h>
namespace kernels {

void generateVoxels(const int batch_size, const int max_num_points, const float *points,
                    const unsigned int *points_size, const float min_x_range, const float max_x_range,
                    const float min_y_range, const float max_y_range, const float min_z_range, const float max_z_range,
                    const float pillar_x_size, const float pillar_y_size, const float pillar_z_size,
                    const int grid_y_size, const int grid_x_size, const int num_point_values,
                    const int max_points_per_voxel, unsigned int *mask, float *voxels, cudaStream_t stream);

void generateBaseFeatures(const int batch_size, const unsigned int *mask, const float *voxels, const int grid_y_size,
                          const int grid_x_size, const int max_pillar_num, const int max_points_per_voxel,
                          const int num_point_values, unsigned int *pillar_num, float *voxel_features,
                          unsigned int *voxel_num_points, unsigned int *coords, cudaStream_t stream);

void generateFeatures(const int batch_size, const float *voxel_features, const unsigned int *coords,
                      const float voxel_x, const float voxel_y, const float voxel_z, const float range_min_x,
                      const float range_min_y, const float range_min_z, const unsigned int voxel_features_size,
                      const unsigned int max_points, const unsigned int max_voxels, const unsigned int num_point_values,
                      unsigned int *voxel_num_points, unsigned int *pillar_nums, float *features, cudaStream_t stream);

void generateVoxelizationFeature(const int batch_size, const int max_num_points, const float *points,
                                 const unsigned int *point_size, const float min_x_range, const float max_x_range,
                                 const float min_y_range, const float max_y_range, const float min_z_range,
                                 const float max_z_range, const float pillar_x_size, const float pillar_y_size,
                                 const float pillar_z_size, const unsigned int num_point_value,
                                 const int max_points_per_voxel, const int max_pillar_num,
                                 const unsigned int num_pillar_feature_value, unsigned int *point_num_voxel,
                                 float *voxel_feature, unsigned int *pillar_num, float *squeeze_voxel_feature,
                                 unsigned int *squeeze_voxel_num_points, unsigned int *pillar_coords,
                                 float *pillar_feature, cudaStream_t stream);

}  // namespace kernels
#endif  // NV_INFRA_VOXELIZATION_CUH
