#pragma once

/**
 * @file memory.h
 * @brief Declares Atlas shared-pointer utilities for host and device memory ownership.
 *
 * @details
 * This header provides a unified ownership interface for objects that may live:
 * - in standard host-managed memory, or
 * - in CUDA-managed/device-accessible memory when `ATLAS_TASKING_CUDA` is enabled.
 *
 * The API exposes:
 * - @ref atlas::host_shared_ptr for ordinary host-side shared ownership,
 * - @ref atlas::device_shared_ptr for CUDA-capable shared ownership,
 * - @ref atlas::make_host_shared for host allocation and construction,
 * - @ref atlas::make_device_shared for device/managed allocation and construction.
 *
 * ## Backend split
 * The implementation differs depending on whether CUDA tasking is enabled:
 * - with `ATLAS_TASKING_CUDA`, Atlas provides a custom CUDA-aware
 *   @ref device_shared_ptr with a manually managed reference count stored in
 *   CUDA managed memory,
 * - otherwise, both host and device shared pointers are simple aliases of
 *   `std::shared_ptr`.
 *
 * ## Design intent
 * This abstraction allows Atlas code to use a consistent shared-ownership model
 * across host-only and CUDA-enabled builds while preserving:
 * - host convenience,
 * - backend portability,
 * - support for objects that must be visible in both host and device code.
 *
 * ## CUDA mode
 * In CUDA-enabled builds:
 * - the managed object and its reference count are allocated with `cudaMallocManaged`,
 * - reference counting is performed through @ref device_refcount,
 * - device-side increments/decrements use CUDA atomics when compiled for device code,
 * - host-side destruction frees both the managed object storage and its reference count.
 *
 * ## Non-CUDA mode
 * In non-CUDA builds:
 * - `device_shared_ptr<T>` is simply an alias to `std::shared_ptr<T>`,
 * - both `make_device_shared` and `make_host_shared` forward to `std::make_shared`.
 *
 * ## Lifetime notes
 * In CUDA mode, the custom device pointer is intended for shared ownership of
 * single objects allocated in managed memory. The implementation assumes that:
 * - the pointee can be constructed with placement new,
 * - host-side destruction is responsible for reclaiming memory,
 * - device-side destruction only participates in reference-count decrementing.
 *
 * ---
 */

#ifdef ATLAS_TASKING_CUDA

#include <cuda_runtime.h>
#include <memory>
#include <type_traits>
#include <utility>

namespace atlas {

#ifdef __CUDA_ARCH__
/**
 * @brief Backend macro for atomic increment on CUDA device code.
 *
 * @details
 * In device compilation mode, this expands to `atomicAdd`.
 */
#define ATLAS_ATOMIC_ADD(addr, val) atomicAdd((addr), (val))

/**
 * @brief Backend macro for atomic decrement on CUDA device code.
 *
 * @details
 * In device compilation mode, this expands to `atomicSub`.
 */
#define ATLAS_ATOMIC_SUB(addr, val) atomicSub((addr), (val))
#else
/**
 * @brief Backend macro for increment in host compilation mode.
 *
 * @details
 * In host compilation mode, this falls back to ordinary scalar addition.
 */
#define ATLAS_ATOMIC_ADD(addr, val) (*(addr) += (val))

/**
 * @brief Backend macro for decrement in host compilation mode.
 *
 * @details
 * In host compilation mode, this falls back to ordinary scalar subtraction.
 */
#define ATLAS_ATOMIC_SUB(addr, val) ((*(addr)) -= (val))
#endif

/**
 * @brief Lightweight shared reference-count handle stored in CUDA-managed memory.
 *
 * @details
 * @ref device_refcount wraps a raw pointer to an integer reference counter and
 * provides small helper functions for:
 * - incrementing the count,
 * - decrementing the count,
 * - querying the current use count.
 *
 * The counter itself is expected to reside in CUDA managed memory so that it is
 * visible to both host and device code.
 *
 * ## Ownership
 * This type does not own the counter memory. It only references it.
 *
 * ## Atomic behavior
 * - In device code, increments and decrements use CUDA atomics.
 * - In host code, the current implementation uses ordinary scalar operations.
 *
 * ---
 */
struct device_refcount {
    /**
     * @brief Pointer to the shared reference counter.
     *
     * @details
     * This pointer is expected to refer to CUDA-managed memory allocated for a
     * single integer reference count.
     */
    int* count;

    /**
     * @brief Default constructor.
     *
     * @details
     * Initializes the handle with no associated counter.
     */
    __host__ __device__
    device_refcount()
        : count(nullptr) { }

    /**
     * @brief Construct a reference-count handle from a raw counter pointer.
     *
     * @param c Pointer to the shared reference count.
     */
    __host__ __device__ explicit device_refcount(int* c)
        : count(c) { }

    /**
     * @brief Increment the reference count if a counter is present.
     *
     * @details
     * This operation is a no-op when @ref count is null.
     */
    __host__ __device__ void
    add_ref() const {
        if (!count) return;
        ATLAS_ATOMIC_ADD(count, 1);
    }

    /**
     * @brief Decrement the reference count and report whether the caller released the last reference.
     *
     * @details
     * This operation is a no-op when @ref count is null.
     *
     * @return `true` if the previous count was `1` and the release consumed the
     *         final reference; otherwise `false`.
     */
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

    /**
     * @brief Return the current reference count.
     *
     * @return Current count value, or `0` if no counter is present.
     */
    __host__ __device__ int
    use_count() const {
        return count ? *count : 0;
    }
};

/**
 * @brief CUDA-aware shared pointer for single objects allocated in managed memory.
 *
 * @details
 * @ref device_shared_ptr is Atlas's custom shared-ownership smart pointer used
 * in CUDA-enabled builds for objects allocated with `cudaMallocManaged`.
 *
 * It stores:
 * - a raw pointer to the managed object,
 * - a @ref device_refcount referencing a managed integer counter.
 *
 * ## Ownership model
 * Multiple @ref device_shared_ptr instances may share ownership of the same
 * managed object. The last host-side owner is responsible for:
 * - running the pointee destructor,
 * - freeing the managed object storage,
 * - freeing the managed reference-count storage.
 *
 * ## Host/device behavior
 * - copy and move operations are available in both host and device code,
 * - device-side destruction only decrements the reference count,
 * - host-side destruction performs full reclamation when the last reference is released.
 *
 * ## Intended use
 * This type is suitable for single objects that need:
 * - shared ownership semantics,
 * - visibility from both host and device code,
 * - construction in managed memory.
 *
 * ---
 *
 * @tparam T Pointee type.
 */
template <typename T>
class device_shared_ptr {
private:
    /**
     * @brief Raw pointer to the managed object.
     */
    T* ptr;

    /**
     * @brief Shared reference-count handle.
     */
    device_refcount ref;

public:
    /**
     * @brief Default constructor.
     *
     * @details
     * Constructs an empty shared pointer.
     */
    __host__ __device__
    device_shared_ptr()
        : ptr(nullptr)
        , ref(nullptr) { }

    /**
     * @brief Construct from a managed object pointer and managed reference counter.
     *
     * @param p Pointer to the managed object.
     * @param c Pointer to the managed reference count.
     */
    __host__ __device__
    device_shared_ptr(T* p, int* c)
        : ptr(p)
        , ref(c) { }

    /**
     * @brief Copy constructor.
     *
     * @details
     * Shares ownership with @p o and increments the reference count when present.
     *
     * @param o Source shared pointer.
     */
    __host__ __device__
    device_shared_ptr(const device_shared_ptr& o)
        : ptr(o.ptr)
        , ref(o.ref) {
        if (ref.count) ref.add_ref();
    }

    /**
     * @brief Move constructor.
     *
     * @details
     * Transfers ownership state from @p o without incrementing the reference count.
     *
     * @param o Source shared pointer.
     */
    __host__ __device__
    device_shared_ptr(device_shared_ptr&& o) noexcept
        : ptr(o.ptr)
        , ref(o.ref) {
        o.ptr       = nullptr;
        o.ref.count = nullptr;
    }

    /**
     * @brief Copy assignment operator.
     *
     * @details
     * Releases the current ownership state, then shares ownership with @p o.
     *
     * @param o Source shared pointer.
     * @return `*this`.
     */
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

    /**
     * @brief Move assignment operator.
     *
     * @details
     * Releases the current ownership state, then transfers ownership from @p o.
     *
     * @param o Source shared pointer.
     * @return `*this`.
     */
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

    /**
     * @brief Destructor.
     *
     * @details
     * - In device code, this decrements the shared reference count.
     * - In host code, this may reclaim the pointee and counter storage if the
     *   last reference is released.
     */
    __host__ __device__ ~device_shared_ptr() {
#ifdef __CUDA_ARCH__
        if (ref.count) ATLAS_ATOMIC_SUB(ref.count, 1);
#else
        release_host_side();
#endif
    }

    /**
     * @brief Return the raw managed object pointer.
     *
     * @return Raw pointee pointer, or `nullptr` for an empty pointer.
     */
    __host__ __device__ T*
    get() const {
        return ptr;
    }

    /**
     * @brief Dereference the managed object.
     *
     * @return Reference to the pointee.
     */
    __host__ __device__ T&
    operator*() const {
        return *ptr;
    }

    /**
     * @brief Access a member of the managed object.
     *
     * @return Raw pointee pointer.
     */
    __host__ __device__ T*
    operator->() const {
        return ptr;
    }

    /**
     * @brief Return the current shared reference count.
     *
     * @return Use count associated with the managed object.
     */
    __host__ __device__ int
    use_count() const {
        return ref.use_count();
    }

    /**
     * @brief Return whether the pointer is non-empty.
     *
     * @return `true` if a managed object is present; otherwise `false`.
     */
    __host__ __device__ explicit
    operator bool() const {
        return ptr != nullptr;
    }

private:
    /**
     * @brief Release the current ownership state from host code.
     *
     * @details
     * This function decrements the host-visible reference count and, when the
     * last reference is released, performs:
     * - explicit destruction of the pointee,
     * - `cudaFree` of the pointee storage,
     * - `cudaFree` of the reference-count storage.
     *
     * It also clears this pointer's local state afterward.
     */
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
};

/**
 * @brief Allocate and construct a shared object in CUDA managed memory.
 *
 * @details
 * This function:
 * - allocates managed storage for an object of type `U`,
 * - allocates managed storage for an integer reference count,
 * - constructs the object in-place with placement new,
 * - initializes the reference count to `1`,
 * - returns a @ref device_shared_ptr owning the result.
 *
 * If allocation fails, any partially allocated storage is freed and an empty
 * @ref device_shared_ptr is returned.
 *
 * @param args Constructor arguments forwarded to the pointee type.
 * @return Shared pointer owning the newly constructed managed object.
 *
 * @tparam T Requested pointee type, decayed internally to `U`.
 * @tparam Args Constructor argument types.
 */
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

/**
 * @brief Construct a host-owned shared object using `std::make_shared`.
 *
 * @param args Constructor arguments forwarded to the pointee type.
 * @return Host shared pointer owning the constructed object.
 *
 * @tparam T Requested pointee type, decayed internally.
 * @tparam Args Constructor argument types.
 */
template <typename T, typename... Args>
inline std::shared_ptr<std::decay_t<T>>
make_host_shared(Args&&... args) {
    using U = std::decay_t<T>;
    return std::make_shared<U>(std::forward<Args>(args)...);
}

/**
 * @brief Convenience alias for ordinary host shared ownership.
 *
 * @tparam T Pointee type.
 */
template <typename T>
using host_shared_ptr = std::shared_ptr<T>;

} // namespace atlas

#else

#include <memory>
#include <type_traits>
#include <utility>

namespace atlas {

/**
 * @brief Convenience alias for host shared ownership in non-CUDA builds.
 *
 * @details
 * In host-only builds, Atlas uses `std::shared_ptr` directly for host-owned objects.
 *
 * @tparam T Pointee type.
 */
template <typename T>
using host_shared_ptr = std::shared_ptr<T>;

/**
 * @brief Convenience alias for device shared ownership in non-CUDA builds.
 *
 * @details
 * In non-CUDA builds, device ownership collapses to ordinary host shared ownership,
 * so this alias is also `std::shared_ptr<T>`.
 *
 * @tparam T Pointee type.
 */
template <typename T>
using device_shared_ptr = std::shared_ptr<T>;

/**
 * @brief Construct a device-shared object in non-CUDA builds.
 *
 * @details
 * Since no CUDA-specific allocation is required, this simply forwards to
 * `std::make_shared`.
 *
 * @param args Constructor arguments forwarded to the pointee type.
 * @return Shared pointer owning the constructed object.
 *
 * @tparam T Requested pointee type, decayed internally.
 * @tparam Args Constructor argument types.
 */
template <typename T, typename... Args>
inline std::shared_ptr<std::decay_t<T>>
make_device_shared(Args&&... args) {
    using U = std::decay_t<T>;
    return std::make_shared<U>(std::forward<Args>(args)...);
}

/**
 * @brief Construct a host-shared object in non-CUDA builds.
 *
 * @param args Constructor arguments forwarded to the pointee type.
 * @return Shared pointer owning the constructed object.
 *
 * @tparam T Requested pointee type, decayed internally.
 * @tparam Args Constructor argument types.
 */
template <typename T, typename... Args>
inline std::shared_ptr<std::decay_t<T>>
make_host_shared(Args&&... args) {
    using U = std::decay_t<T>;
    return std::make_shared<U>(std::forward<Args>(args)...);
}

} // namespace atlas

#endif