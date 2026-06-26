#pragma once

#include <atlas/buffer/device_buffer.h>
#include <atlas/fluid/fluid.h>
#include <atlas/math/math.h>
#include <atlas/memory/memory.h>
#include <atlas/universe/universe.h>

#include <cstdint>
#include <type_traits>

namespace atlas {

template <typename T>
class Searcher {
    static_assert(std::is_floating_point_v<T>, "Searcher requires a floating-point T");

public:
    Searcher() = default;

    ATLAS_HOST ATLAS_FORCE_INLINE
    Searcher(UniverseHostPtr<T> universe, FluidHostPtr<T> fluid);

    virtual ~Searcher() = default;

    Searcher(const Searcher&) = default;
    Searcher&
    operator=(const Searcher&)
        = default;
    Searcher(Searcher&&) noexcept = default;
    Searcher&
    operator=(Searcher&&) noexcept = default;

    ATLAS_HOST virtual void
    build()
        = 0;

    ATLAS_HOST ATLAS_FORCE_INLINE virtual void
    invalidate() noexcept;

    ATLAS_HOST ATLAS_FORCE_INLINE virtual void
    reset() noexcept;

    ATLAS_HOST ATLAS_NODISCARD ATLAS_FORCE_INLINE virtual Vector3<T>
    lower_corner() const noexcept;

    ATLAS_HOST ATLAS_NODISCARD ATLAS_FORCE_INLINE virtual Vector3<int>
    grid_size() const noexcept;

    ATLAS_HOST ATLAS_NODISCARD ATLAS_FORCE_INLINE virtual T
    inverse_cell_size() const noexcept;

    ATLAS_HOST ATLAS_NODISCARD ATLAS_FORCE_INLINE virtual T
    cell_size() const noexcept;

    ATLAS_HOST ATLAS_NODISCARD ATLAS_FORCE_INLINE virtual const int*
    indices() const noexcept;

    ATLAS_HOST ATLAS_NODISCARD ATLAS_FORCE_INLINE virtual const int*
    cell_start() const noexcept;

    ATLAS_HOST ATLAS_NODISCARD ATLAS_FORCE_INLINE virtual const int*
    cell_end() const noexcept;

    ATLAS_HOST ATLAS_NODISCARD ATLAS_FORCE_INLINE virtual const int*
    neighbor_offsets() const noexcept;

    ATLAS_HOST ATLAS_NODISCARD ATLAS_FORCE_INLINE virtual const int*
    neighbor_indices() const noexcept;

    ATLAS_HOST ATLAS_NODISCARD ATLAS_FORCE_INLINE virtual int
    neighbor_count() const noexcept;

    ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE static std::uint32_t
    linear_key(int ix, int iy, int iz, const Vector3<int>& gs) noexcept;

    ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE static std::uint32_t
    linear_key(const Vector3<int>& cell, const Vector3<int>& gs) noexcept;

    ATLAS_ALL_DEVICE ATLAS_NODISCARD ATLAS_FORCE_INLINE static Vector3<int>
    cell_for(const Vector3<T>& position,
             const Vector3<T>& lower_corner,
             T inverse_cell_size,
             const Vector3<int>& grid_size) noexcept;

    ATLAS_ALL_DEVICE ATLAS_NODISCARD ATLAS_FORCE_INLINE static bool
    contains_cell(const Vector3<int>& cell, const Vector3<int>& grid_size) noexcept;

    ATLAS_ALL_DEVICE ATLAS_NODISCARD ATLAS_FORCE_INLINE static int
    search_radius_for(T length, T cell_size) noexcept;

protected:
    ATLAS_HOST ATLAS_FORCE_INLINE void
    validate_dependencies(const char* owner) const;

    ATLAS_HOST ATLAS_FORCE_INLINE const Vector3<T>*
    position_ptr() const noexcept;

    ATLAS_HOST ATLAS_FORCE_INLINE int
    active_count() const noexcept;

    ATLAS_HOST ATLAS_FORCE_INLINE void
    prepare_grid_buffers(int alive);

public:
    ATLAS_HOST ATLAS_FORCE_INLINE void
    init_indices_iota(int alive);

    ATLAS_HOST ATLAS_FORCE_INLINE void
    compute_grid_keys(int alive, const Vector3<T>* positions);

protected:
    ATLAS_HOST ATLAS_FORCE_INLINE void
    sort_by_key(int alive);

public:
    ATLAS_HOST ATLAS_FORCE_INLINE void
    build_cell_ranges(int alive);

protected:
    template <typename CandidateFilter>
    ATLAS_HOST ATLAS_FORCE_INLINE void
    build_cell_neighbors(int alive, const Vector3<T>* positions, CandidateFilter filter);

    ATLAS_HOST ATLAS_FORCE_INLINE int
    finalize_neighbor_offsets(int alive);

protected:
    ATLAS_HOST ATLAS_FORCE_INLINE void
    clear_neighbors();

protected:
    UniverseHostPtr<T> _universe {};

    FluidHostPtr<T> _fluid {};

    DeviceBuffer<std::uint32_t> _keys;

    DeviceBuffer<int> _indices;

    DeviceBuffer<int> _cell_start;

    DeviceBuffer<int> _cell_end;

    DeviceBuffer<int> _neighbor_offsets;

    DeviceBuffer<int> _neighbor_indices;

    DeviceBuffer<int> _neighbor_counts;

    DeviceBuffer<int> _neighbor_total_count;

    int _neighbor_count {};

    bool _is_invalidated { true };
};

}

namespace atlas {
template <typename T>
using SearcherHostPtr = atlas::host_shared_ptr<Searcher<T>>;

template <typename T>
using SearcherDevicePtr = atlas::device_shared_ptr<Searcher<T>>;

}

#include <atlas/searcher/searcher.hpp>