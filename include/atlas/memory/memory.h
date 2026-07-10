#pragma once

/**
 * @file memory.h
 * @brief Smart-pointer aliases and a CUDA-managed, device-capturable @c shared_ptr.
 *
 * The whole header has two shapes selected by @c ATLAS_TASKING_CUDA:
 *   - When CUDA tasking is on, @c device_shared_ptr is a hand-rolled reference-counted
 *     pointer over @c cudaMallocManaged storage, so the *same* pointer object can be
 *     captured by value into a device lambda and dereferenced on the GPU while its
 *     refcount stays coherent across host and device via atomics.
 *   - Otherwise (CPU-only build) every "device" alias collapses to the corresponding
 *     @c std::shared_ptr / @c std::unique_ptr, so callers compile unchanged.
 */

#ifdef ATLAS_TASKING_CUDA

#include <cuda_runtime.h>
#include <memory>
#include <type_traits>
#include <utility>

namespace atlas {

#ifdef __CUDA_ARCH__

/**
 * @def ATLAS_ATOMIC_ADD
 * @brief Atomically add @p val to the integer at @p addr.
 *
 * In device code (@c __CUDA_ARCH__ defined) this expands to CUDA's @c atomicAdd so the
 * managed refcount stays correct under concurrent access from many threads; in host code
 * it degrades to a plain non-atomic @c += (host mutation is assumed single-threaded).
 */
#define ATLAS_ATOMIC_ADD(addr, val) atomicAdd((addr), (val))

/**
 * @def ATLAS_ATOMIC_SUB
 * @brief Atomically subtract @p val from the integer at @p addr; see @c ATLAS_ATOMIC_ADD.
 */
#define ATLAS_ATOMIC_SUB(addr, val) atomicSub((addr), (val))
#else

/**
 * @def ATLAS_ATOMIC_ADD
 * @brief Host fallback: non-atomic in-place add (the device build uses @c atomicAdd).
 */
#define ATLAS_ATOMIC_ADD(addr, val) (*(addr) += (val))

/**
 * @def ATLAS_ATOMIC_SUB
 * @brief Host fallback: non-atomic in-place subtract (device build uses @c atomicSub).
 */
#define ATLAS_ATOMIC_SUB(addr, val) ((*(addr)) -= (val))
#endif

/**
 * @brief Non-owning handle to a shared reference counter living in managed memory.
 *
 * Wraps a raw @c int* that points at a single @c cudaMallocManaged integer holding the
 * strong count for a @c device_shared_ptr. It performs the atomic bump/decrement but
 * never allocates or frees the counter itself — that is the owning smart pointer's job.
 * A null @c count means "no counter" (a default-constructed / moved-from pointer) and
 * every operation is a safe no-op.
 */
struct device_refcount {

    int* count; ///< Managed-memory strong counter, or nullptr when there is none.

    /**
     * @brief Construct an empty handle referring to no counter.
     */
    __host__ __device__
    device_refcount()
        : count(nullptr) { }

    /**
     * @brief Construct a handle over an existing managed counter.
     * @param c Pointer to a @c cudaMallocManaged @c int, or nullptr.
     */
    __host__ __device__ explicit device_refcount(int* c)
        : count(c) { }

    /**
     * @brief Atomically increment the strong count (no-op when empty).
     *
     * @note @c const because it mutates only the pointee counter, not the handle.
     */
    __host__ __device__ void
    add_ref() const {
        if (!count) return;
        ATLAS_ATOMIC_ADD(count, 1);
    }

    /**
     * @brief Atomically decrement the strong count and report whether it hit zero.
     *
     * @return @c true when this call dropped the last reference (pre-decrement value was
     *         1), meaning the caller now owns the teardown; @c false otherwise, including
     *         the empty-counter case.
     * @note On the device @c atomicSub returns the old value directly; on the host the
     *       old value is read first, then subtracted, to reproduce the same semantics.
     * @warning Currently unused by @c device_shared_ptr, which inlines its own
     *          decrement logic (see @c release_host_side and the destructor).
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
     * @brief Current strong reference count.
     * @return The counter's value, or 0 when there is no counter. Not synchronized; on
     *         the device the value may be stale the instant it is read.
     */
    __host__ __device__ int
    use_count() const {
        return count ? *count : 0;
    }
};

/**
 * @brief Reference-counted owning pointer whose object lives in CUDA managed memory.
 *
 * Both the payload @p T and its strong counter are allocated with
 * @c cudaMallocManaged (see @c make_device_shared), so a copy of this pointer can be
 * captured by value into a device lambda and the object dereferenced on the GPU. The
 * refcount is maintained with @c ATLAS_ATOMIC_ADD/SUB, which are true atomics on the
 * device and plain arithmetic on the host.
 *
 * Teardown is intentionally asymmetric between host and device:
 *   - On the **host**, dropping the count to zero runs @c ~T() and @c cudaFree on both
 *     the object and the counter (see @c release_host_side).
 *   - On the **device**, destructors/assignments only decrement the counter; they never
 *     @c cudaFree (illegal from a kernel). Managed allocations are therefore reclaimed by
 *     the host copy that outlives the device work, not by a device-side drop to zero.
 *
 * @tparam T Managed payload type; must be destructible.
 * @warning The host teardown is not thread-safe: @c release_host_side reads the counter,
 *          branches, then frees, without a lock. Concurrent host releases of the last
 *          references could race. Device execution is expected to have been synchronized
 *          before the owning host pointer is destroyed.
 */
template <typename T>
class device_shared_ptr {
private:
    T* ptr; ///< Managed-memory payload, or nullptr when empty/moved-from.

    device_refcount ref; ///< Handle to the shared managed strong counter.

public:
    /**
     * @brief Construct an empty pointer owning nothing (null object, null counter).
     */
    __host__ __device__
    device_shared_ptr()
        : ptr(nullptr)
        , ref(nullptr) { }

    /**
     * @brief Adopt an already-allocated managed object and its counter.
     *
     * Takes ownership of @p p and @p c without bumping the count — the count is expected
     * to already reflect this pointer (as set up by @c make_device_shared, which stores 1).
     *
     * @param p Managed-memory object pointer.
     * @param c Managed-memory strong counter pointer.
     */
    __host__ __device__
    device_shared_ptr(T* p, int* c)
        : ptr(p)
        , ref(c) { }

    /**
     * @brief Share ownership with @p o, incrementing the strong count.
     * @param o Source pointer to copy; left unchanged.
     */
    __host__ __device__
    device_shared_ptr(const device_shared_ptr& o)
        : ptr(o.ptr)
        , ref(o.ref) {
        if (ref.count) ref.add_ref();
    }

    /**
     * @brief Steal ownership from @p o without touching the count.
     * @param o Source pointer; left empty (null object and counter) afterwards.
     */
    __host__ __device__
    device_shared_ptr(device_shared_ptr&& o) noexcept
        : ptr(o.ptr)
        , ref(o.ref) {
        o.ptr       = nullptr;
        o.ref.count = nullptr;
    }

    /**
     * @brief Copy-assign: release the current reference, then share @p o's.
     *
     * The old reference is dropped before adopting the new one. On the device this is a
     * bare atomic decrement (no free); on the host it runs the full @c release_host_side
     * teardown that may @c cudaFree. Self-assignment is a no-op.
     *
     * @param o Source pointer to share.
     * @return @c *this.
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
     * @brief Move-assign: release the current reference, then steal @p o's.
     *
     * Like the copy assignment for the release half, but adopts @p o's reference without
     * a count bump and leaves @p o empty. Self-assignment is a no-op.
     *
     * @param o Source pointer; left empty afterwards.
     * @return @c *this.
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
     * @brief Drop this reference; free the managed object and counter on the last host drop.
     *
     * Device path only decrements the counter (freeing from a kernel is illegal); the
     * host path delegates to @c release_host_side, which frees when the count reaches zero.
     */
    __host__ __device__ ~device_shared_ptr() {
#ifdef __CUDA_ARCH__
        if (ref.count) ATLAS_ATOMIC_SUB(ref.count, 1);
#else
        release_host_side();
#endif
    }

    /**
     * @brief Raw pointer to the managed object.
     * @return The object address (dereferenceable on both host and device), or nullptr.
     */
    __host__ __device__ T*
    get() const {
        return ptr;
    }

    /**
     * @brief Dereference the managed object.
     * @return Reference to the object. Undefined behavior when empty.
     */
    __host__ __device__ T&
    operator*() const {
        return *ptr;
    }

    /**
     * @brief Member access on the managed object.
     * @return The object pointer for @c -> chaining. Undefined behavior when empty.
     */
    __host__ __device__ T*
    operator->() const {
        return ptr;
    }

    /**
     * @brief Current strong reference count (0 when empty). Not synchronized on device.
     * @return The counter value; see @c device_refcount::use_count.
     */
    __host__ __device__ int
    use_count() const {
        return ref.use_count();
    }

    /**
     * @brief Test whether this pointer owns an object.
     * @return @c true when the payload is non-null. Explicit to avoid accidental
     *         conversions.
     */
    __host__ __device__ explicit
    operator bool() const {
        return ptr != nullptr;
    }

private:
    /**
     * @brief Host-only teardown: decrement the count and free on the last reference.
     *
     * Runs @c ~T() and @c cudaFree on both the object and the counter once the strong
     * count reaches zero, then nulls this pointer's fields so a later destructor is inert.
     * Host-only because @c cudaFree cannot be called from device code.
     *
     * @note The leading @c current==0 branch treats an already-zero counter as "free now"
     *       — a defensive path for a counter that is somehow at zero while this pointer
     *       still holds it; see the review note in the unit report.
     * @warning Not thread-safe (read-branch-free without a lock).
     */
    void
    release_host_side() {
        if (!ref.count) return;

        // Defensive: a counter already at zero is treated as "nobody else holds it",
        // so free the object and counter immediately.
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

        // Normal path: drop our reference; free only when we were the last holder.
        int old = *ref.count;
        ATLAS_ATOMIC_SUB(ref.count, 1);

        if (old == 1) {
            if (ptr) {
                ptr->~T();
                cudaFree(ptr);
            }
            cudaFree(ref.count);
        }

        // Detach either way so a subsequent destructor call is a no-op.
        ptr       = nullptr;
        ref.count = nullptr;
    }
};

/**
 * @brief Allocate a @c T in CUDA managed memory and wrap it in a @c device_shared_ptr.
 *
 * Allocates the object and its strong counter with @c cudaMallocManaged, in-place
 * constructs the object from @p args, and seeds the count to 1. Managed storage is why
 * the returned pointer (and copies of it) can be dereferenced from a device lambda.
 *
 * @tparam T Requested type; decayed to @c U before allocation.
 * @tparam Args Constructor argument types, perfectly forwarded.
 * @param args Arguments forwarded to @c U's constructor.
 * @return An owning pointer to the new managed object, or an **empty** pointer if either
 *         @c cudaMallocManaged failed (both partial allocations are freed first). Callers
 *         must check the result rather than assume success.
 * @note The object is placement-@c new'd; on the last host release @c ~U() plus
 *       @c cudaFree reclaim it (see @c device_shared_ptr::release_host_side).
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
        // Roll back a half-succeeded allocation so nothing leaks on failure.
        if (obj) cudaFree(obj);
        if (cnt) cudaFree(cnt);
        return device_shared_ptr<U>();
    }

    new (obj) U(std::forward<Args>(args)...);
    *cnt = 1;

    return device_shared_ptr<U>(obj, cnt);
}

/**
 * @brief Allocate a host-only @c T via @c std::make_shared.
 *
 * Distinct from @c make_device_shared: the object lives in ordinary host memory and must
 * not be touched from device code.
 *
 * @tparam T Requested type; decayed before allocation.
 * @tparam Args Constructor argument types, perfectly forwarded.
 * @param args Arguments forwarded to the constructor.
 * @return A @c std::shared_ptr owning the new host object.
 */
template <typename T, typename... Args>
inline std::shared_ptr<std::decay_t<T>>
make_host_shared(Args&&... args) {
    using U = std::decay_t<T>;
    return std::make_shared<U>(std::forward<Args>(args)...);
}

/**
 * @brief Owning, shared, host-only pointer. Alias for @c std::shared_ptr.
 * @tparam T Pointee type.
 */
template <typename T>
using host_shared_ptr = std::shared_ptr<T>;

/**
 * @brief Owning, exclusive, host-only pointer. Alias for @c std::unique_ptr.
 * @tparam T Pointee type.
 */
template <typename T>
using host_unique_ptr = std::unique_ptr<T>;

/**
 * @brief Allocate a host-only @c T via @c std::make_unique.
 * @tparam T Requested type; decayed before allocation.
 * @tparam Args Constructor argument types, perfectly forwarded.
 * @param args Arguments forwarded to the constructor.
 * @return A @c std::unique_ptr owning the new host object.
 */
template <typename T, typename... Args>
inline std::unique_ptr<std::decay_t<T>>
make_host_unique(Args&&... args) {
    using U = std::decay_t<T>;
    return std::make_unique<U>(std::forward<Args>(args)...);
}

}

#else

#include <memory>
#include <type_traits>
#include <utility>

namespace atlas {

// CPU-only build: every pointer flavor is a plain std::shared_ptr / std::unique_ptr, so
// code written against the aliases below compiles and runs identically without CUDA.

/**
 * @brief Owning, shared, host-only pointer. Alias for @c std::shared_ptr.
 * @tparam T Pointee type.
 */
template <typename T>
using host_shared_ptr = std::shared_ptr<T>;

/**
 * @brief CPU-only stand-in for the managed device pointer: a plain @c std::shared_ptr.
 *
 * Without CUDA there is no device memory, so "device" ownership collapses to ordinary
 * host ownership; the alias keeps call sites source-compatible with the CUDA build.
 *
 * @tparam T Pointee type.
 */
template <typename T>
using device_shared_ptr = std::shared_ptr<T>;

/**
 * @brief Owning, exclusive, host-only pointer. Alias for @c std::unique_ptr.
 * @tparam T Pointee type.
 */
template <typename T>
using host_unique_ptr = std::unique_ptr<T>;

/**
 * @brief CPU-only @c make_device_shared: allocates on the host via @c std::make_shared.
 * @tparam T Requested type; decayed before allocation.
 * @tparam Args Constructor argument types, perfectly forwarded.
 * @param args Arguments forwarded to the constructor.
 * @return A @c std::shared_ptr owning the new object.
 */
template <typename T, typename... Args>
inline std::shared_ptr<std::decay_t<T>>
make_device_shared(Args&&... args) {
    using U = std::decay_t<T>;
    return std::make_shared<U>(std::forward<Args>(args)...);
}

/**
 * @brief Allocate a host object via @c std::make_shared.
 * @tparam T Requested type; decayed before allocation.
 * @tparam Args Constructor argument types, perfectly forwarded.
 * @param args Arguments forwarded to the constructor.
 * @return A @c std::shared_ptr owning the new object.
 */
template <typename T, typename... Args>
inline std::shared_ptr<std::decay_t<T>>
make_host_shared(Args&&... args) {
    using U = std::decay_t<T>;
    return std::make_shared<U>(std::forward<Args>(args)...);
}

/**
 * @brief Allocate a host object via @c std::make_unique.
 * @tparam T Requested type; decayed before allocation.
 * @tparam Args Constructor argument types, perfectly forwarded.
 * @param args Arguments forwarded to the constructor.
 * @return A @c std::unique_ptr owning the new object.
 */
template <typename T, typename... Args>
inline std::unique_ptr<std::decay_t<T>>
make_host_unique(Args&&... args) {
    using U = std::decay_t<T>;
    return std::make_unique<U>(std::forward<Args>(args)...);
}

}

#endif