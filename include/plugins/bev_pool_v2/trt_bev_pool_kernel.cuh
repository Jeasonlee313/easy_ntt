#ifndef TRT_BEV_POOL_KERNEL_CUH
#define TRT_BEV_POOL_KERNEL_CUH
#include <cuda_runtime.h>

namespace nvinfer1 {
namespace plugin {
// CUDA function declarations
void bev_pool_v2(int c, int n_intervals, const float *depth, const float *feat, const int *ranks_depth,
                 const int *ranks_feat, const int *ranks_bev, const int *interval_starts, const int *interval_lengths,
                 float *out, cudaStream_t stream);

void bev_pool_v2_set_zero(int n_points, float *out);
}  // namespace plugin
}  // namespace nvinfer1
#endif  // TRT_BEV_POOL_KERNEL_CUH
