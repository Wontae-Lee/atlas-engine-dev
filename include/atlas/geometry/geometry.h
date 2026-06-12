#pragma once

#include <atlas/geometry/geometry_type.h>
#include <atlas/math/math.h>
#include <atlas/memory/memory.h>
#include <atlas/spatial/axis_aligned_bounding_box.h>
#include <type_traits>

namespace atlas {

template <typename T>
struct GeometryOperator;

template <typename T>
class DeviceGeometryViewFactory {
public:
    DeviceGeometryViewFactory()          = default;
    virtual ~DeviceGeometryViewFactory() = default;

    ATLAS_HOST ATLAS_NODISCARD virtual GeometryOperator<T>
    make_device_geometry_view() const = 0;
};

template <typename T>
class Geometry {
    static_assert(std::is_floating_point_v<T>, "Geometry requires a floating-point T");

public:
    Geometry() = default;

    virtual ~Geometry() = default;

    ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE virtual atlas::Vector<T, 3>
    closest_point(const atlas::Vector<T, 3>& p) const noexcept = 0;

    ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE virtual atlas::Vector<T, 3>
    closest_normal(const atlas::Vector<T, 3>& p) const noexcept = 0;

    ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE virtual T
    signed_distance(const atlas::Vector<T, 3>& p) const noexcept = 0;

    ATLAS_ALL_DEVICE ATLAS_NODISCARD ATLAS_FORCE_INLINE virtual bool
    is_inside(const atlas::Vector<T, 3>& p, T tolerance) const noexcept = 0;

    ATLAS_ALL_DEVICE ATLAS_NODISCARD ATLAS_FORCE_INLINE virtual bool
    is_on_surface(const atlas::Vector<T, 3>& p, T tolerance) const noexcept = 0;

    ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE virtual atlas::Vector<T, 3>
    centroid() const noexcept = 0;

    ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE virtual atlas::AxisAlignedBoundingBox<T>
    bound() const noexcept = 0;

    ATLAS_ALL_DEVICE ATLAS_NODISCARD ATLAS_FORCE_INLINE virtual bool
    is_valid() const noexcept = 0;

    ATLAS_ALL_DEVICE ATLAS_NODISCARD ATLAS_FORCE_INLINE virtual GeometryType
    type() const noexcept = 0;
};

template <typename T>
ATLAS_HOST ATLAS_NODISCARD GeometryOperator<T>
make_device_geometry_view(const Geometry<T>& geometry);

template <typename T>
using GeometryHostPtr = atlas::host_shared_ptr<Geometry<T>>;

template <typename T>
using GeometryDevicePtr = atlas::device_shared_ptr<Geometry<T>>;

}