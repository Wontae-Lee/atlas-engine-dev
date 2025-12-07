#ifndef INCLUDE_ATLAS_GEOMETRY_SURFACE_H
#define INCLUDE_ATLAS_GEOMETRY_SURFACE_H

#include <atlas/math/math.h>
#include <atlas/spatial/axis_aligned_bounding_box.h>
#include <atlas/spatial/ray.h>

namespace atlas ::geometry {

template <typename T, typename Derived>
class Surface {
public:
    ATLAS_NODISCARD ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE bool
    intersects(const Ray<T>& r) const {
        return d().intersects(r);
    }

    ATLAS_NODISCARD ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE AABB<T>
    bound() const {
        return d().bound();
    }

    ATLAS_NODISCARD ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE T
    signed_distance(const Vector3<T>& p) const {
        return d().signed_distance(p);
    }

    ATLAS_NODISCARD ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE Vector3<T>
    closest_point(const Vector3<T>& p) const {
        return d().closest_point(p);
    }

    ATLAS_NODISCARD ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE Vector3<T>
    closest_normal(const Vector3<T>& p) const {
        return d().closest_normal(p);
    }

    ATLAS_NODISCARD ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE T
    closest_distance(const Vector3<T>& p) const {
        return d().closest_distance(p);
    }

    ATLAS_NODISCARD ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE bool
    is_inside(const Vector3<T>& p) const {
        return d().is_inside(p);
    }

protected:
    ATLAS_NODISCARD ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE const Derived&
    d() const { return static_cast<const Derived&>(*this); }
    ATLAS_NODISCARD ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE Derived&
    d() { return static_cast<Derived&>(*this); }
};

}

#endif