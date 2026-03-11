#pragma once

/**
 * @file memory.h
 * @brief Unified host/device smart-pointer helpers (CUDA managed allocations or std::shared_ptr fallback).
 *
 * @details
 * This header provides a minimal abstraction for shared ownership of objects across
 * Atlas host code and (optionally) CUDA device code.
 *
 * Two build modes are supported:
 *
 * ## CUDA build (`ATLAS_TASKING_CUDA` defined)
 * - `atlas::device_shared_ptr<T>` is a lightweight, device-usable shared pointer that
 *   holds:
 *   - a managed pointer to an object (`cudaMallocManaged`)
 *   - a managed pointer to an `int` reference count (`cudaMallocManaged`)
 * - Reference counting on device uses CUDA atomics (`atomicAdd/atomicSub`).
 * - Host-side destruction frees the managed object and the refcount when the last owner releases it.
 *
 * ## CPU-only build
 * - `atlas::device_shared_ptr<T>` is an alias to `std::shared_ptr<T>`.
 * - `make_device_shared` / `make_host_shared` both return `std::shared_ptr`.
 *
 * This design keeps higher-level code backend-agnostic while still allowing simple
 * device-friendly ownership semantics when CUDA is enabled.
 *
 * @warning
 * This implementation is intentionally lightweight and has important constraints:
 * - The CUDA `device_shared_ptr` does **not** delete/free memory on the device; it only
 *   decrements the counter. Memory reclamation is performed on the host path.
 * - Cross-device-thread lifetime management can be subtle. Prefer treating
 *   `device_shared_ptr` as an object handle whose lifetime is ultimately governed by host code.
 */

#ifdef ATLAS_TASKING_CUDA

#include <cuda_runtime.h>
#include <memory>
#include <type_traits>
#include <utility>

namespace atlas {

// -----------------------------------------------------------------------------
// Atomic helpers
// -----------------------------------------------------------------------------

/**
 * @brief Atomic add/sub wrappers that compile in both host and device code.
 *
 * @details
 * - In device code (`__CUDA_ARCH__`), use CUDA atomics.
 * - In host code, fall back to non-atomic increments/decrements.
 *
 * @note
 * Host-side fallback is not thread-safe; it is only intended for simple ownership
 * transitions on the host side (typical std::shared_ptr-like usage).
 */
#ifdef __CUDA_ARCH__
#define ATLAS_ATOMIC_ADD(addr, val) atomicAdd((addr), (val))
#define ATLAS_ATOMIC_SUB(addr, val) atomicSub((addr), (val))
#else
#define ATLAS_ATOMIC_ADD(addr, val) (*(addr) += (val))
#define ATLAS_ATOMIC_SUB(addr, val) ((*(addr)) -= (val))
#endif

// -----------------------------------------------------------------------------
// device_refcount
// -----------------------------------------------------------------------------

/**
 * @brief Device-visible reference counter wrapper.
 *
 * @details
 * Holds a pointer to an `int` counter allocated in CUDA managed memory.
 * On device, updates are performed using atomic operations.
 *
 * @note
 * - `count == nullptr` represents an empty/invalid state.
 * - `release()` returns true if this call releases the last reference (i.e., transitions 1 -> 0).
 */
struct device_refcount {
    int* count;

    /// @brief Constructs an empty refcount (null counter).
    __host__ __device__
    device_refcount()
        : count(nullptr) {}

    /// @brief Constructs from an existing counter pointer.
    __host__ __device__
    explicit device_refcount(int* c)
        : count(c) {}

    /// @brief Increments the reference count if valid.
    __host__ __device__
    void add_ref() const {
        if (!count) return;
        ATLAS_ATOMIC_ADD(count, 1);
    }

    /**
     * @brief Decrements the reference count if valid.
     *
     * @return `true` if the reference count reached zero as a result of this call.
     *
     * @note
     * On device, the return value is computed using the value returned by `atomicSub`.
     */
    __host__ __device__
    bool release() const {
        if (!count) return false;
#ifdef __CUDA_ARCH__
        // atomicSub returns the old value
        int old = ATLAS_ATOMIC_SUB(count, 1);
        return (old == 1);
#else
        // Host fallback: preserve the old value for last-owner detection
        int old = *count;
        ATLAS_ATOMIC_SUB(count, 1);
        return (old == 1);
#endif
    }

    /// @brief Returns the current reference count, or 0 for null.
    __host__ __device__
    int use_count() const {
        return count ? *count : 0;
    }
};

// -----------------------------------------------------------------------------
// device_shared_ptr
// -----------------------------------------------------------------------------

/**
 * @brief Minimal shared-ownership pointer usable in CUDA device code.
 *
 * @details
 * Stores:
 * - `ptr`: managed pointer to the object (`cudaMallocManaged`)
 * - `ref`: managed refcount pointer (also managed memory)
 *
 * Copying increments the refcount; destruction decrements it.
 *
 * @note
 * - In device code, the destructor only decrements the counter (no freeing).
 * - In host code, `release_host_side()` will destroy/free when this instance
 *   was the last owner.
 *
 * @tparam T Object type.
 */
template <typename T>
class device_shared_ptr {
private:
    T* ptr;            ///< Managed pointer to object storage.
    device_refcount ref; ///< Reference-count wrapper.

public:
    /// @brief Constructs an empty pointer.
    __host__ __device__
    device_shared_ptr()
        : ptr(nullptr), ref(nullptr) {}

    /// @brief Constructs from raw object pointer and raw counter pointer.
    __host__ __device__
    device_shared_ptr(T* p, int* c)
        : ptr(p), ref(c) {}

    /// @brief Copy constructor increments the reference count.
    __host__ __device__
    device_shared_ptr(const device_shared_ptr& o)
        : ptr(o.ptr), ref(o.ref) {
        if (ref.count) ref.add_ref();
    }

    /// @brief Move constructor transfers ownership without changing refcount.
    __host__ __device__
    device_shared_ptr(device_shared_ptr&& o) noexcept
        : ptr(o.ptr), ref(o.ref) {
        o.ptr       = nullptr;
        o.ref.count = nullptr;
    }

    /// @brief Copy assignment releases current and then shares ownership of `o`.
    __host__ __device__
    device_shared_ptr& operator=(const device_shared_ptr& o) {
        if (this != &o) {
#ifdef __CUDA_ARCH__
            // Device side: just decrement current refcount; no deletion on device.
            if (ref.count) ATLAS_ATOMIC_SUB(ref.count, 1);
#else
            // Host side: destroy/free if we were last owner.
            release_host_side();
#endif
            ptr = o.ptr;
            ref = o.ref;
            if (ref.count) ref.add_ref();
        }
        return *this;
    }

    /// @brief Move assignment releases current and then takes ownership from `o`.
    __host__ __device__
    device_shared_ptr& operator=(device_shared_ptr&& o) noexcept {
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

    /// @brief Destructor decrements refcount; host side may destroy/free the allocation.
    __host__ __device__
    ~device_shared_ptr() {
#ifdef __CUDA_ARCH__
        if (ref.count) ATLAS_ATOMIC_SUB(ref.count, 1);
#else
        release_host_side();
#endif
    }

    /// @brief Returns the raw pointer (may be null).
    __host__ __device__
    T* get() const { return ptr; }

    /// @brief Dereferences the stored pointer (undefined behavior if null).
    __host__ __device__
    T& operator*() const { return *ptr; }

    /// @brief Member access (undefined behavior if null).
    __host__ __device__
    T* operator->() const { return ptr; }

    /// @brief Returns current reference count.
    __host__ __device__
    int use_count() const { return ref.use_count(); }

    /// @brief Checks whether this pointer is non-null.
    __host__ __device__
    explicit operator bool() const { return ptr != nullptr; }

private:
    /**
     * @brief Host-side release path that destroys/frees on last reference.
     *
     * @details
     * This path is only compiled/used on the host. It performs:
     * - decrement the refcount
     * - if old count was 1, destroy object and `cudaFree` both object and counter
     *
     * @note
     * The `current == 0` early branch handles edge cases where the counter was
     * externally manipulated or already zero; it attempts to clean up to avoid leaks.
     */
    void release_host_side() {
        if (!ref.count) return;

        // Snapshot the current count. If it's already 0, attempt cleanup defensively.
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

        // Decrement and decide whether we were the last owner.
        int old = *ref.count;
        ATLAS_ATOMIC_SUB(ref.count, 1);

        if (old == 1) {
            if (ptr) {
                ptr->~T();
                cudaFree(ptr);
            }
            cudaFree(ref.count);
        }

        // Clear this handle regardless.
        ptr       = nullptr;
        ref.count = nullptr;
    }
};

// -----------------------------------------------------------------------------
// Factory helpers
// -----------------------------------------------------------------------------

/**
 * @brief Allocates an object in CUDA managed memory and returns a `device_shared_ptr`.
 *
 * @details
 * - Allocates managed memory for both the object and an `int` reference counter.
 * - Constructs the object in-place using placement `new`.
 * - Initializes the reference count to 1.
 *
 * @tparam T    Object type (decayed).
 * @tparam Args Constructor argument pack.
 * @param args  Arguments forwarded to `T`'s constructor.
 * @return A `device_shared_ptr<U>` owning the managed allocation, or empty on allocation failure.
 *
 * @note
 * - Requires `T` to be destructible. A conservative static_assert is used.
 * - Allocation failures return an empty `device_shared_ptr`.
 */
template <typename T, typename... Args>
inline device_shared_ptr<std::decay_t<T>>
make_device_shared(Args&&... args) {
    using U = std::decay_t<T>;
    static_assert(std::is_trivially_destructible<U>::value == false || std::is_destructible<U>::value,
                  "Type must be destructible");

    U* obj   = nullptr;
    int* cnt = nullptr;

    // Allocate in managed memory to make the pointer visible to both host and device.
    cudaError_t e1 = cudaMallocManaged(&obj, sizeof(U), cudaMemAttachGlobal);
    cudaError_t e2 = cudaMallocManaged(&cnt, sizeof(int), cudaMemAttachGlobal);

    if (e1 != cudaSuccess || e2 != cudaSuccess) {
        if (obj) cudaFree(obj);
        if (cnt) cudaFree(cnt);
        return device_shared_ptr<U>();
    }

    // Construct the object and initialize the refcount.
    new (obj) U(std::forward<Args>(args)...);
    *cnt = 1;

    return device_shared_ptr<U>(obj, cnt);
}

/**
 * @brief Constructs a host-only shared pointer using `std::make_shared`.
 */
template <typename T, typename... Args>
inline std::shared_ptr<std::decay_t<T>>
make_host_shared(Args&&... args) {
    using U = std::decay_t<T>;
    return std::make_shared<U>(std::forward<Args>(args)...);
}

/**
 * @brief Host-side shared pointer alias.
 */
template <typename T>
using host_shared_ptr = std::shared_ptr<T>;

} // namespace atlas

#else // -----------------------------------------------------------------------
// CPU-only fallback
// -----------------------------------------------------------------------

#include <memory>
#include <type_traits>
#include <utility>

namespace atlas {

/**
 * @brief Host-side shared pointer alias (CPU-only build).
 */
template <typename T>
using host_shared_ptr = std::shared_ptr<T>;

/**
 * @brief Device shared pointer alias (CPU-only build).
 *
 * @details
 * In non-CUDA builds, "device" pointers are treated as host pointers, so we
 * reuse `std::shared_ptr`.
 */
template <typename T>
using device_shared_ptr = std::shared_ptr<T>;

/**
 * @brief CPU-only implementation of `make_device_shared` (maps to `std::make_shared`).
 */
template <typename T, typename... Args>
inline std::shared_ptr<std::decay_t<T>>
make_device_shared(Args&&... args) {
    using U = std::decay_t<T>;
    return std::make_shared<U>(std::forward<Args>(args)...);
}

/**
 * @brief CPU-only implementation of `make_host_shared` (maps to `std::make_shared`).
 */
template <typename T, typename... Args>
inline std::shared_ptr<std::decay_t<T>>
make_host_shared(Args&&... args) {
    using U = std::decay_t<T>;
    return std::make_shared<U>(std::forward<Args>(args)...);
}

} // namespace atlas

#endif
