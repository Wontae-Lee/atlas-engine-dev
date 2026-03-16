#pragma once

#include <algorithm>
#include <cmath>
#include <limits>

namespace atlas::system::detail {

template <typename T>
ATLAS_HOST ATLAS_FORCE_INLINE bool
is_finite_vector(const Vector3<T>& v) noexcept {
    return std::isfinite(v.x) && std::isfinite(v.y) && std::isfinite(v.z);
}

template <typename T, typename Engine>
ATLAS_HOST ATLAS_FORCE_INLINE T
uniform_real(Engine& engine, T min_value = T(0), T max_value = T(1)) {
    atlas::uniform_real_distribution<T> dist(min_value, max_value);
    return dist(engine);
}

template <typename T, typename Engine>
ATLAS_HOST ATLAS_FORCE_INLINE Vector3<T>
sample_unit_direction(Engine& engine) {
    const T z       = uniform_real<T>(engine, T(-1), T(1));
    const T phi     = uniform_real<T>(engine, T(0), T(2) * static_cast<T>(M_PI));
    const T radial2 = std::max(T(0), T(1) - z * z);
    const T radial  = static_cast<T>(std::sqrt(radial2));
    return Vector3<T>(radial * static_cast<T>(std::cos(phi)),
                      radial * static_cast<T>(std::sin(phi)),
                      z);
}

template <typename T, typename Engine>
ATLAS_HOST ATLAS_FORCE_INLINE Vector3<T>
sample_disk(const Vector3<T>& center,
            const Vector3<T>& tangent,
            const Vector3<T>& bitangent,
            const T radius,
            Engine& engine) {
    const T r   = radius * static_cast<T>(std::sqrt(uniform_real<T>(engine)));
    const T phi = uniform_real<T>(engine, T(0), T(2) * static_cast<T>(M_PI));
    return center + tangent * (r * static_cast<T>(std::cos(phi)))
        + bitangent * (r * static_cast<T>(std::sin(phi)));
}

template <typename T, typename Engine>
ATLAS_HOST ATLAS_FORCE_INLINE Vector3<T>
sample_box_volume(const atlas::geometry::Box<T>& box, Engine& engine) {
    return Vector3<T>(
        uniform_real<T>(engine, box.lower_corner.x, box.upper_corner.x),
        uniform_real<T>(engine, box.lower_corner.y, box.upper_corner.y),
        uniform_real<T>(engine, box.lower_corner.z, box.upper_corner.z));
}

template <typename T, typename Engine>
ATLAS_HOST ATLAS_FORCE_INLINE Vector3<T>
sample_box_surface(const atlas::geometry::Box<T>& box, Engine& engine) {
    const T dx = box.upper_corner.x - box.lower_corner.x;
    const T dy = box.upper_corner.y - box.lower_corner.y;
    const T dz = box.upper_corner.z - box.lower_corner.z;

    const T yz_area = dy * dz;
    const T xz_area = dx * dz;
    const T xy_area = dx * dy;
    const T total   = T(2) * (yz_area + xz_area + xy_area);

    if (total <= T(0)) return box.lower_corner;

    T pick = uniform_real<T>(engine, T(0), total);
    auto sample_y = [&]() { return uniform_real<T>(engine, box.lower_corner.y, box.upper_corner.y); };
    auto sample_z = [&]() { return uniform_real<T>(engine, box.lower_corner.z, box.upper_corner.z); };
    auto sample_x = [&]() { return uniform_real<T>(engine, box.lower_corner.x, box.upper_corner.x); };

    if ((pick -= yz_area) <= T(0)) return Vector3<T>(box.lower_corner.x, sample_y(), sample_z());
    if ((pick -= yz_area) <= T(0)) return Vector3<T>(box.upper_corner.x, sample_y(), sample_z());
    if ((pick -= xz_area) <= T(0)) return Vector3<T>(sample_x(), box.lower_corner.y, sample_z());
    if ((pick -= xz_area) <= T(0)) return Vector3<T>(sample_x(), box.upper_corner.y, sample_z());
    if ((pick -= xy_area) <= T(0)) return Vector3<T>(sample_x(), sample_y(), box.lower_corner.z);
    return Vector3<T>(sample_x(), sample_y(), box.upper_corner.z);
}

template <typename T, typename Engine>
ATLAS_HOST ATLAS_FORCE_INLINE Vector3<T>
sample_sphere_surface(const atlas::geometry::Sphere<T>& sphere, Engine& engine) {
    return sphere.center + sample_unit_direction<T>(engine) * sphere.radius;
}

template <typename T, typename Engine>
ATLAS_HOST ATLAS_FORCE_INLINE Vector3<T>
sample_sphere_volume(const atlas::geometry::Sphere<T>& sphere, Engine& engine) {
    const T radius = sphere.radius * static_cast<T>(std::cbrt(uniform_real<T>(engine)));
    return sphere.center + sample_unit_direction<T>(engine) * radius;
}

template <typename T, typename Engine>
ATLAS_HOST ATLAS_FORCE_INLINE Vector3<T>
sample_cylinder_volume(const atlas::geometry::Cylinder<T>& cylinder, Engine& engine) {
    const T phi    = uniform_real<T>(engine, T(0), T(2) * static_cast<T>(M_PI));
    const T radial = cylinder.radius * static_cast<T>(std::sqrt(uniform_real<T>(engine)));
    const T z      = uniform_real<T>(engine, -cylinder.height * T(0.5), cylinder.height * T(0.5));
    return Vector3<T>(
        cylinder.center.x + radial * static_cast<T>(std::cos(phi)),
        cylinder.center.y + radial * static_cast<T>(std::sin(phi)),
        cylinder.center.z + z);
}

template <typename T, typename Engine>
ATLAS_HOST ATLAS_FORCE_INLINE Vector3<T>
sample_cylinder_surface(const atlas::geometry::Cylinder<T>& cylinder, Engine& engine) {
    const T side_area = T(2) * static_cast<T>(M_PI) * cylinder.radius * cylinder.height;
    const T cap_area  = static_cast<T>(M_PI) * cylinder.radius * cylinder.radius;
    const T total     = side_area + T(2) * cap_area;

    if (total <= T(0)) return cylinder.center;

    const T pick = uniform_real<T>(engine, T(0), total);
    if (pick < side_area) {
        const T phi = uniform_real<T>(engine, T(0), T(2) * static_cast<T>(M_PI));
        const T z   = uniform_real<T>(engine, -cylinder.height * T(0.5), cylinder.height * T(0.5));
        return Vector3<T>(
            cylinder.center.x + cylinder.radius * static_cast<T>(std::cos(phi)),
            cylinder.center.y + cylinder.radius * static_cast<T>(std::sin(phi)),
            cylinder.center.z + z);
    }

    const T z = (uniform_real<T>(engine) < T(0.5)) ? (-cylinder.height * T(0.5)) : (cylinder.height * T(0.5));
    return sample_disk<T>(
        Vector3<T>(cylinder.center.x, cylinder.center.y, cylinder.center.z + z),
        Vector3<T>(T(1), T(0), T(0)),
        Vector3<T>(T(0), T(1), T(0)),
        cylinder.radius,
        engine);
}

template <typename T, typename Engine>
ATLAS_HOST ATLAS_FORCE_INLINE Vector3<T>
sample_triangle_surface(const atlas::geometry::Triangle<T>& triangle, Engine& engine) {
    const T u      = uniform_real<T>(engine);
    const T v      = uniform_real<T>(engine);
    const T su     = static_cast<T>(std::sqrt(u));
    const T w0     = T(1) - su;
    const T w1     = su * (T(1) - v);
    const T w2     = su * v;
    return triangle.a * w0 + triangle.b * w1 + triangle.c * w2;
}

template <typename T, typename Engine>
ATLAS_HOST ATLAS_FORCE_INLINE Vector3<T>
sample_plane_surface(const atlas::geometry::Plane<T>& plane,
                     const T patch_extent,
                     Engine& engine) {
    const T n2 = plane.normal.length_squared();
    if (n2 <= atlas::eps) return Vector3<T>(T(0), T(0), T(0));

    const T inv_n_len = T(1) / static_cast<T>(std::sqrt(n2));
    const Vector3<T> n = plane.normal * inv_n_len;

    // n · x = d. Scaling n by inv_n_len requires the representative point to use the original normal.
    const Vector3<T> origin = plane.normal * (plane.offset / n2);

    Vector3<T> tangent, bitangent;
    atlas::random::build_orthonormal_basis(n, tangent, bitangent);

    return origin
        + tangent * uniform_real<T>(engine, -patch_extent, patch_extent)
        + bitangent * uniform_real<T>(engine, -patch_extent, patch_extent);
}

template <typename T>
ATLAS_HOST ATLAS_FORCE_INLINE T
triangle_area(const TriangleContainer4<T>& tri) {
    return T(0.5) * atlas::math::cross(tri.b() - tri.a(), tri.c() - tri.a()).length();
}

template <typename T, typename Engine>
ATLAS_HOST ATLAS_FORCE_INLINE Vector3<T>
sample_triangle_mesh_surface(const atlas::geometry::TriangleMesh<T>& mesh,
                             const HostBuffer<T>& prefix,
                             const T total_area,
                             Engine& engine) {
    if (mesh.triangles.empty()) return Vector3<T>(T(0), T(0), T(0));
    if (total_area <= T(0)) return mesh.triangles.front().a();

    const T pick = uniform_real<T>(engine, T(0), total_area);
    const auto it = std::lower_bound(prefix.begin(), prefix.end(), pick);
    const std::size_t tri_index = static_cast<std::size_t>(std::distance(prefix.begin(), it));
    const auto& tri = mesh.triangles[std::min(tri_index, mesh.triangles.size() - 1)];

    const T u      = uniform_real<T>(engine);
    const T v      = uniform_real<T>(engine);
    const T su     = static_cast<T>(std::sqrt(u));
    const T w0     = T(1) - su;
    const T w1     = su * (T(1) - v);
    const T w2     = su * v;
    return tri.a() * w0 + tri.b() * w1 + tri.c() * w2;
}

template <typename T>
ATLAS_HOST ATLAS_FORCE_INLINE bool
has_finite_bounds(const atlas::spatial::AxisAlignedBoundingBox<T>& bounds) noexcept {
    return is_finite_vector(bounds.lower_corner)
        && is_finite_vector(bounds.upper_corner)
        && bounds.width() >= T(0)
        && bounds.height() >= T(0)
        && bounds.depth() >= T(0);
}

template <typename T, typename Engine>
ATLAS_HOST ATLAS_FORCE_INLINE Vector3<T>
sample_aabb(const atlas::spatial::AxisAlignedBoundingBox<T>& bounds, Engine& engine) {
    return Vector3<T>(
        uniform_real<T>(engine, bounds.lower_corner.x, bounds.upper_corner.x),
        uniform_real<T>(engine, bounds.lower_corner.y, bounds.upper_corner.y),
        uniform_real<T>(engine, bounds.lower_corner.z, bounds.upper_corner.z));
}

template <typename T, typename Engine>
ATLAS_HOST ATLAS_FORCE_INLINE HostBuffer<Vector3<T>>
sample_mesh_volume_rejection(const atlas::geometry::TriangleMesh<T>& mesh,
                             std::size_t count,
                             std::size_t max_attempt_multiplier,
                             Engine& engine) {
    HostBuffer<Vector3<T>> out;
    out.reserve(count);

    const auto bounds = mesh.bound();
    if (!has_finite_bounds<T>(bounds)) return out;

    const std::size_t max_attempts = std::max<std::size_t>(count, 1) * std::max<std::size_t>(max_attempt_multiplier, 1);
    std::size_t attempts = 0;

    while (out.size() < count && attempts < max_attempts) {
        const Vector3<T> p = sample_aabb<T>(bounds, engine);
        if (mesh.signed_distance(p) <= T(0)) out.push_back(p);
        ++attempts;
    }

    return out;
}

} // namespace atlas::system::detail

namespace atlas::system {

template <typename T>
SpawnOperator<T>::SpawnOperator(const SpawnType type_,
                                const std::uint32_t seed_,
                                const T plane_surface_extent_) noexcept
    : type(type_)
    , seed(seed_)
    , plane_surface_extent(plane_surface_extent_) {
}

template <typename T>
HostBuffer<Vector3<T>>
SpawnOperator<T>::spawn(const atlas::geometry::Geometry<T>& geometry, const std::size_t count) const {
    HostBuffer<Vector3<T>> out;
    out.reserve(count);

    if (count == 0 || !geometry.is_valid()) return out;

    atlas::default_random_engine<T> engine(seed);

    switch (geometry.type()) {
        case atlas::geometry::GeometryType::Box: {
            const auto& box = static_cast<const atlas::geometry::Box<T>&>(geometry);
            for (std::size_t i = 0; i < count; ++i) {
                out.push_back(type == SpawnType::Surface
                                  ? detail::sample_box_surface<T>(box, engine)
                                  : detail::sample_box_volume<T>(box, engine));
            }
            return out;
        }

        case atlas::geometry::GeometryType::Sphere: {
            const auto& sphere = static_cast<const atlas::geometry::Sphere<T>&>(geometry);
            for (std::size_t i = 0; i < count; ++i) {
                out.push_back(type == SpawnType::Surface
                                  ? detail::sample_sphere_surface<T>(sphere, engine)
                                  : detail::sample_sphere_volume<T>(sphere, engine));
            }
            return out;
        }

        case atlas::geometry::GeometryType::Cylinder: {
            const auto& cylinder = static_cast<const atlas::geometry::Cylinder<T>&>(geometry);
            for (std::size_t i = 0; i < count; ++i) {
                out.push_back(type == SpawnType::Surface
                                  ? detail::sample_cylinder_surface<T>(cylinder, engine)
                                  : detail::sample_cylinder_volume<T>(cylinder, engine));
            }
            return out;
        }

        case atlas::geometry::GeometryType::Triangle: {
            if (type == SpawnType::Volume) return out;
            const auto& triangle = static_cast<const atlas::geometry::Triangle<T>&>(geometry);
            for (std::size_t i = 0; i < count; ++i) {
                out.push_back(detail::sample_triangle_surface<T>(triangle, engine));
            }
            return out;
        }

        case atlas::geometry::GeometryType::TriangleMesh: {
            const auto& mesh = static_cast<const atlas::geometry::TriangleMesh<T>&>(geometry);
            if (type == SpawnType::Volume) {
                return detail::sample_mesh_volume_rejection<T>(mesh, count, max_volume_rejection_iterations, engine);
            }

            HostBuffer<T> prefix(mesh.triangles.size(), T(0));
            T total_area = T(0);
            for (std::size_t i = 0; i < mesh.triangles.size(); ++i) {
                total_area += detail::triangle_area(mesh.triangles[i]);
                prefix[i] = total_area;
            }

            for (std::size_t i = 0; i < count; ++i) {
                out.push_back(detail::sample_triangle_mesh_surface<T>(mesh, prefix, total_area, engine));
            }
            return out;
        }

        case atlas::geometry::GeometryType::Plane: {
            if (type == SpawnType::Volume) return out;
            const auto& plane = static_cast<const atlas::geometry::Plane<T>&>(geometry);
            for (std::size_t i = 0; i < count; ++i) {
                out.push_back(detail::sample_plane_surface<T>(plane, plane_surface_extent, engine));
            }
            return out;
        }
    }

    return out;
}

template <typename T>
HostBuffer<Vector3<T>>
SpawnOperator<T>::spawn(const atlas::host_shared_ptr<atlas::geometry::Geometry<T>>& geometry, const std::size_t count) const {
    if (!geometry) return {};
    return spawn(*geometry, count);
}

template <typename T>
HostBuffer<Vector3<T>>
SpawnOperator<T>::operator()(const atlas::geometry::Geometry<T>& geometry, const std::size_t count) const {
    return spawn(geometry, count);
}

template <typename T>
HostBuffer<Vector3<T>>
SpawnOperator<T>::operator()(const atlas::host_shared_ptr<atlas::geometry::Geometry<T>>& geometry, const std::size_t count) const {
    return spawn(geometry, count);
}

} // namespace atlas::system
