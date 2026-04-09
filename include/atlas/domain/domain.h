#pragma once

#include <atlas/buffer/device_buffer.h>
#include <atlas/geometry/geometry.h>
#include <atlas/math/math.h>
#include <atlas/memory/memory.h>
#include <atlas/memory/raw_pointer_cast.h>

#include <optional>

namespace atlas::system {

enum class DomainType : int {
    isothermal,
    variable
};

template <typename T>
struct DomainDeviceProbe {

    DomainType type = DomainType::variable;

    T* field_temperature = nullptr;

    Vector3<T>* field_force = nullptr;

    Vector3<T> lower_corner;

    Vector3<T> upper_corner;

    Vector3<int> grid_size { 1, 1, 1 };

    T cell_size = T(1);

    T cell_volume = T(1);

    T inv_h = T(1);

    int num_of_cells = 0;
};

template <typename T>
class Domain {
public:
    class Builder;

    Domain() = delete;

    Domain(const Vector3<T>& lower_corner,
           const Vector3<T>& upper_corner,
           T cell_size,
           DomainType type = DomainType::variable,
           std::optional<T> temperature = std::nullopt);

    ~Domain() = default;

    ATLAS_HOST ATLAS_FORCE_INLINE static Builder
    builder() noexcept;

    ATLAS_HOST ATLAS_FORCE_INLINE DomainDeviceProbe<T>
    make_device_probe() noexcept;

    ATLAS_HOST ATLAS_NODISCARD ATLAS_FORCE_INLINE int
    number_of_cells() const noexcept;

    ATLAS_HOST ATLAS_NODISCARD ATLAS_FORCE_INLINE DomainType
    type() const noexcept;

    ATLAS_HOST ATLAS_NODISCARD ATLAS_FORCE_INLINE T
    isothermal_field_temperature() const;

    ATLAS_HOST ATLAS_NODISCARD ATLAS_FORCE_INLINE Vector3<T>
    lower_corner() const noexcept;

    ATLAS_HOST ATLAS_NODISCARD ATLAS_FORCE_INLINE Vector3<T>
    upper_corner() const noexcept;

    ATLAS_HOST ATLAS_NODISCARD ATLAS_FORCE_INLINE Vector3<int>
    grid_size() const noexcept;

    ATLAS_HOST ATLAS_NODISCARD ATLAS_FORCE_INLINE T
    cell_size() const noexcept;

    ATLAS_HOST ATLAS_NODISCARD ATLAS_FORCE_INLINE T
    cell_volume() const noexcept;

    ATLAS_HOST ATLAS_NODISCARD ATLAS_FORCE_INLINE T
    inverse_cell_size() const noexcept;

private:
    DeviceBuffer<T> d_field_temperature;

    DeviceBuffer<Vector3<T>> d_field_force;

    Vector3<T> _lower_corner;

    Vector3<T> _upper_corner;

    Vector3<int> _grid_size { 1, 1, 1 };

    T _cell_size = T(1);

    T _cell_volume = T(1);

    T _inv_h = T(1);

    int _num_of_cells = 1;

    DomainType _type = DomainType::variable;

    std::optional<T> _isothermal_field_temperature;

    std::uint64_t _probe_count = 0;
};

template <typename T>
class Domain<T>::Builder final {
public:
    Builder() = default;

    ATLAS_HOST ATLAS_FORCE_INLINE Domain<T>
    build() const;

    ATLAS_HOST ATLAS_FORCE_INLINE atlas::host_shared_ptr<Domain<T>>
    make_host_shared() const;

    ATLAS_HOST ATLAS_FORCE_INLINE Builder&
    with_geometry(const GeometryHostPtr<T>& geometry) noexcept;

    ATLAS_HOST ATLAS_FORCE_INLINE Builder&
    with_lower_corner(const Vector3<T>& v) noexcept;

    ATLAS_HOST ATLAS_FORCE_INLINE Builder&
    with_upper_corner(const Vector3<T>& v) noexcept;

    ATLAS_HOST ATLAS_FORCE_INLINE Builder&
    with_cell_size(T h) noexcept;

    ATLAS_HOST ATLAS_FORCE_INLINE Builder&
    with_type(DomainType type) noexcept;

    ATLAS_HOST ATLAS_FORCE_INLINE Builder&
    with_temperature(T temperature) noexcept;

private:
    void
    validate() const;

private:
    Vector3<T> _lower_corner { T(0), T(0), T(0) };

    Vector3<T> _upper_corner { T(1), T(1), T(1) };

    T _cell_size = T(1);

    DomainType _type = DomainType::variable;

    std::optional<T> _isothermal_field_temperature;
};

}

namespace atlas {

using DomainType = atlas::system::DomainType;

template <typename T>
using Domain = atlas::system::Domain<T>;
template <typename T>
using DomainHostPtr = atlas::host_shared_ptr<atlas::system::Domain<T>>;
template <typename T>
using DomainDevicePtr = atlas::device_shared_ptr<atlas::system::Domain<T>>;
}

#include <atlas/domain/domain.hpp>
