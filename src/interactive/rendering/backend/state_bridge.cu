#include "rendering/backend/state_bridge.h"

#include "rendering/opengl/buffer.h"

#include <cuda_gl_interop.h>
#include <cuda_runtime_api.h>

#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>

namespace atlas::interactive {

namespace {

void
check_cuda(const cudaError_t result, const char* operation) {
    if (result == cudaSuccess) return;
    throw std::runtime_error(std::string(operation) + ": " + cudaGetErrorString(result));
}

}

struct StateBridge::Impl {
    std::unordered_map<unsigned int, cudaGraphicsResource*> resources;
    std::vector<std::byte> staging;
    bool interop_available = true;
};

StateBridge::StateBridge()
    : _impl(std::make_unique<Impl>()) {}

StateBridge::~StateBridge() {
    for (const auto& [buffer, resource] : _impl->resources) {
        static_cast<void>(buffer);
        if (resource != nullptr) cudaGraphicsUnregisterResource(resource);
    }
}

void
StateBridge::release(opengl::Buffer& target) {
    const auto found = _impl->resources.find(target.id());
    if (found == _impl->resources.end()) return;
    check_cuda(cudaGraphicsUnregisterResource(found->second),
               "CUDA/OpenGL buffer unregistration failed");
    _impl->resources.erase(found);
}

void
StateBridge::upload_raw(const void* source,
                        const std::size_t bytes,
                        opengl::Buffer& target) {
    if (bytes == 0) {
        target.allocate(0);
        return;
    }

    const bool grows = bytes > target.capacity();
    if (grows && target.id() != 0) {
        const auto found = _impl->resources.find(target.id());
        if (found != _impl->resources.end()) {
            check_cuda(cudaGraphicsUnregisterResource(found->second),
                       "CUDA/OpenGL buffer unregistration failed");
            _impl->resources.erase(found);
        }
    }

    target.allocate(bytes);
    opengl::Buffer::unbind();

    if (_impl->interop_available) {
        auto [entry, inserted] = _impl->resources.try_emplace(target.id(), nullptr);
        if (inserted) {
            const cudaError_t result = cudaGraphicsGLRegisterBuffer(
                &entry->second,
                target.id(),
                cudaGraphicsRegisterFlagsWriteDiscard);
            if (result != cudaSuccess) {
                _impl->resources.erase(entry);
                _impl->interop_available = false;
                static_cast<void>(cudaGetLastError());
            }
        }

        if (_impl->interop_available) {
            cudaGraphicsResource* resource = entry->second;
            check_cuda(cudaGraphicsMapResources(1, &resource), "CUDA/OpenGL buffer mapping failed");

            void* mapped = nullptr;
            std::size_t mapped_bytes = 0;
            const cudaError_t pointer_result =
                cudaGraphicsResourceGetMappedPointer(&mapped, &mapped_bytes, resource);
            if (pointer_result != cudaSuccess || mapped == nullptr || mapped_bytes < bytes) {
                cudaGraphicsUnmapResources(1, &resource);
                if (pointer_result != cudaSuccess) {
                    check_cuda(pointer_result, "CUDA/OpenGL mapped pointer lookup failed");
                }
                throw std::runtime_error(
                    "CUDA/OpenGL mapped buffer is smaller than the requested upload.");
            }

            const cudaError_t copy_result = cudaMemcpy(mapped, source, bytes, cudaMemcpyDeviceToDevice);
            const cudaError_t unmap_result = cudaGraphicsUnmapResources(1, &resource);
            check_cuda(copy_result, "CUDA/OpenGL device-to-device copy failed");
            check_cuda(unmap_result, "CUDA/OpenGL buffer unmapping failed");
            return;
        }
    }

    _impl->staging.resize(bytes);
    check_cuda(cudaMemcpy(_impl->staging.data(), source, bytes, cudaMemcpyDeviceToHost),
               "CUDA-to-host staging copy failed");
    target.upload(_impl->staging.data(), bytes);
    opengl::Buffer::unbind();
}

}
