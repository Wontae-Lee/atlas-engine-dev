#pragma once
#include <atlas/core/macros.h>
#include <cuda_runtime.h>
namespace atlas {
template <typename T>
ATLAS_HOST void
copy_device_to_host(const T* src, T* dst, std::size_t count) {
    const std::size_t bytes = count * sizeof(T);
    cudaError_t err         = cudaMemcpy(dst, src, bytes, cudaMemcpyDeviceToHost);
    if (err != cudaSuccess) {
        fprintf(stderr,
                "cudaMemcpy (device to host) failed: %s\n",
                cudaGetErrorString(err));
    }
}
}