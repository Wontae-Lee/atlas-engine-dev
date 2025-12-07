#ifndef INCLUDE_ATLAS_MEMORY_MEMORY_H
#define INCLUDE_ATLAS_MEMORY_MEMORY_H

#ifdef ATLAS_TASKING_CUDA

#include <cuda_runtime.h>
#include <memory>
#include <type_traits>
#include <utility>

namespace atlas {

#ifdef __CUDA_ARCH__
#define ATLAS_ATOMIC_ADD(addr, val) atomicAdd((addr), (val))
#define ATLAS_ATOMIC_SUB(addr, val) atomicSub((addr), (val))
#else

#define ATLAS_ATOMIC_ADD(addr, val) (*(addr) += (val))
#define ATLAS_ATOMIC_SUB(addr, val) ((*(addr)) -= (val))
#endif

struct device_refcount {
    int* count;

    __host__ __device__
    device_refcount()
        : count(nullptr) { }

    __host__ __device__ explicit device_refcount(int* c)
        : count(c) { }

    __host__ __device__ void
    add_ref() const {
        if (!count) return;
        ATLAS_ATOMIC_ADD(count, 1);
    }

    __host__ __device__ bool
    release() const {
        if (!count) return false;
#ifdef __CUDA_ARCH__
        int old = ATLAS_ATOMIC_SUB(count, 1);
        return (old == 1);
#else
        int old = *count;
        ATLAS_ATOMIC_SUB(count, 1);
        return (old == 1);
#endif
    }

    __host__ __device__ int
    use_count() const {
        return count ? *count : 0;
    }
};

template <typename T>
class device_shared_ptr {
private:
    T* ptr;
    device_refcount ref;

public:
    __host__ __device__
    device_shared_ptr()
        : ptr(nullptr)
        , ref(nullptr) { }

    __host__ __device__
    device_shared_ptr(T* p, int* c)
        : ptr(p)
        , ref(c) { }

    __host__ __device__
    device_shared_ptr(const device_shared_ptr& o)
        : ptr(o.ptr)
        , ref(o.ref) {
        if (ref.count) ref.add_ref();
    }

    __host__ __device__
    device_shared_ptr(device_shared_ptr&& o) noexcept
        : ptr(o.ptr)
        , ref(o.ref) {
        o.ptr       = nullptr;
        o.ref.count = nullptr;
    }

    __host__ __device__ device_shared_ptr&
    operator=(const device_shared_ptr& o) {
        if (this != &o) {
#ifdef __CUDA_ARCH__

            if (ref.count) ATLAS_ATOMIC_SUB(ref.count, 1);
#else

            release_host_side();
#endif
            ptr = o.ptr;
            ref = o.ref;
            if (ref.count) ref.add_ref();
        }
        return *this;
    }

    __host__ __device__ device_shared_ptr&
    operator=(device_shared_ptr&& o) noexcept {
        if (this != &o) {
#ifdef __CUDA_ARCH__
            if (ref.count) ATLAS_ATOMIC_SUB(ref.count, 1);
#else
            release_host_side();
#endif
            ptr         = o.ptr;
            ref         = o.ref;
            o.ptr       = nullptr;
            o.ref.count = nullptr;
        }
        return *this;
    }

    __host__ __device__ ~device_shared_ptr() {
#ifdef __CUDA_ARCH__

        if (ref.count) ATLAS_ATOMIC_SUB(ref.count, 1);
#else
        release_host_side();
#endif
    }

    __host__ __device__ T*
    get() const {
        return ptr;
    }
    __host__ __device__ T&
    operator*() const {
        return *ptr;
    }
    __host__ __device__ T*
    operator->() const {
        return ptr;
    }
    __host__ __device__ int
    use_count() const {
        return ref.use_count();
    }
    __host__ __device__ explicit
    operator bool() const {
        return ptr != nullptr;
    }

private:
#ifndef __CUDA_ARCH__

    void
    release_host_side() {
        if (!ref.count) return;

        int current = *ref.count;

        if (current == 0) {

            if (ptr) {
                ptr->~T();
                cudaFree(ptr);
            }
            cudaFree(ref.count);
            ptr       = nullptr;
            ref.count = nullptr;
            return;
        }

        int old = *ref.count;
        ATLAS_ATOMIC_SUB(ref.count, 1);

        if (old == 1) {
            if (ptr) {
                ptr->~T();
                cudaFree(ptr);
            }
            cudaFree(ref.count);
        }

        ptr       = nullptr;
        ref.count = nullptr;
    }
#endif
};

template <typename T, typename... Args>
inline device_shared_ptr<std::decay_t<T>>
make_device_shared(Args&&... args) {
    using U = std::decay_t<T>;
    static_assert(std::is_trivially_destructible<U>::value == false || std::is_destructible<U>::value,
                  "Type must be destructible");

    U* obj   = nullptr;
    int* cnt = nullptr;

    cudaError_t e1 = cudaMallocManaged(&obj, sizeof(U), cudaMemAttachGlobal);
    cudaError_t e2 = cudaMallocManaged(&cnt, sizeof(int), cudaMemAttachGlobal);
    if (e1 != cudaSuccess || e2 != cudaSuccess) {
        if (obj) cudaFree(obj);
        if (cnt) cudaFree(cnt);
        return device_shared_ptr<U>();
    }

    new (obj) U(std::forward<Args>(args)...);
    *cnt = 1;

    return device_shared_ptr<U>(obj, cnt);
}
template <typename T, typename... Args>
inline std::shared_ptr<std::decay_t<T>>
make_host_shared(Args&&... args) {
    using U = std::decay_t<T>;
    return std::make_shared<U>(std::forward<Args>(args)...);
}

template <typename T>
using host_shared_ptr = std::shared_ptr<T>;

}

#else

#include <memory>

namespace atlas {

template <typename T>
using host_shared_ptr = std::shared_ptr<T>;

template <typename T>
using device_shared_ptr = std::shared_ptr<T>;

template <typename T, typename... Args>
inline std::shared_ptr<std::decay_t<T>>
make_device_shared(Args&&... args) {
    using U = std::decay_t<T>;
    return std::make_shared<U>(std::forward<Args>(args)...);
}

template <typename T, typename... Args>
inline std::shared_ptr<std::decay_t<T>>
make_host_shared(Args&&... args) {
    using U = std::decay_t<T>;
    return std::make_shared<U>(std::forward<Args>(args)...);
}

}

#endif

#endif