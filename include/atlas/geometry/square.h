#pragma once

#include <atlas/core/macros.h>
#include <atlas/math/math.h>
#include <atlas/memory/memory.h>
#include <atlas/spatial/axis_aligned_bounding_box.h>
#include <atlas/spatial/ray.h>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <limits>

namespace atlas {

class Square final {
public:
    class Builder;

public:
    Float3 center     = Float3(0.0f, 0.0f, 0.0f);
    Float3 normal     = Float3(0.0f, 0.0f, 1.0f);
    float side_length = 1.0f;

    Square() noexcept = default;

    ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE
    Square(const Float3& center_,
           const Float3& normal_,
           const float side_length_) noexcept
        : center(center_)
        , normal(normal_)
        , side_length(side_length_) { }

    Square(const Square& other) noexcept = default;
    Square(Square&& other) noexcept      = default;
    Square&
    operator=(const Square& other) noexcept = default;
    Square&
    operator=(Square&& other) noexcept = default;

    ~Square() noexcept = default;

    ATLAS_NODISCARD ATLAS_HOST static Builder
    builder() noexcept;

    ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE Float3
    closest_point(const Float3& p) const noexcept {
        if (!(side_length > 0.0f)) {
            return p;
        }

        Float3 unit_normal;
        Float3 tangent;
        Float3 bitangent;

        if (!build_basis(normal, unit_normal, tangent, bitangent)) {
            return p;
        }

        const float half_side        = side_length * 0.5f;
        const Float3 center_to_point = p - center;

        const float signed_plane_offset = center_to_point.dot(unit_normal);
        const Float3 projected_point    = p - unit_normal * signed_plane_offset;

        const Float3 planar_offset = projected_point - center;

        const float u = std::clamp(planar_offset.dot(tangent), -half_side, half_side);
        const float v = std::clamp(planar_offset.dot(bitangent), -half_side, half_side);

        return center + tangent * u + bitangent * v;
    }

    ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE Float3
    closest_normal(const Float3&) const noexcept {
        return atlas::normalized_or(normal, Float3(0.0f, 0.0f, 1.0f));
    }

    ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE float
    signed_distance(const Float3& p) const noexcept {
        const Float3 projected_closest_point = closest_point(p);

        const Float3 unit_normal = closest_normal(p);

        const float distance_magnitude = (p - projected_closest_point).length();

        const float sign_test = (p - center).dot(unit_normal);

        return (sign_test >= 0.0f) ? distance_magnitude : -distance_magnitude;
    }

    ATLAS_NODISCARD ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE bool
    is_inside(const Float3& p, const float tolerance = 0.0f) const noexcept {
        if (!(side_length > 0.0f)) {
            return false;
        }

        Float3 unit_normal;
        Float3 tangent;
        Float3 bitangent;

        if (!build_basis(normal, unit_normal, tangent, bitangent)) {
            return false;
        }

        const float half_side        = side_length * 0.5f;
        const Float3 center_to_point = p - center;

        const float signed_plane_offset = center_to_point.dot(unit_normal);

        const float u = center_to_point.dot(tangent);
        const float v = center_to_point.dot(bitangent);

        return std::abs(signed_plane_offset) <= tolerance
            && std::abs(u) <= half_side + tolerance
            && std::abs(v) <= half_side + tolerance;
    }

    ATLAS_NODISCARD ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE bool
    is_on_surface(const Float3& p, const float tolerance = 0.0f) const noexcept {
        return is_inside(p, tolerance);
    }

    ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE Float3
    centroid() const noexcept {
        return center;
    }

    ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE AABB
    bound() const noexcept {
        if (!(side_length > 0.0f)) {
            return AABB(center, center);
        }

        Float3 unit_normal;
        Float3 tangent;
        Float3 bitangent;

        if (!build_basis(normal, unit_normal, tangent, bitangent)) {
            return AABB(center, center);
        }

        const float half_side = side_length * 0.5f;

        const Float3 extent = (atlas::abs(tangent) + atlas::abs(bitangent)) * half_side;

        return AABB(center - extent, center + extent);
    }

    ATLAS_NODISCARD ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE bool
    is_valid() const noexcept {
        return atlas::isfinite(center)
            && atlas::isfinite(normal)
            && normal.length_squared() > 0.0f
            && atlas::isfinite(side_length)
            && side_length > 0.0f;
    }

    ATLAS_NODISCARD ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE HitSurface
    trace(const Ray& ray) const noexcept {
        HitSurface hit {};

        if (!is_valid()) {
            return hit;
        }

        Float3 unit_normal;
        Float3 tangent;
        Float3 bitangent;

        if (!build_basis(normal, unit_normal, tangent, bitangent)) {
            return hit;
        }

        float distance;

        if (!atlas::ray_plane_distance(center, unit_normal, ray, distance)) {
            return hit;
        }

        const float epsilon = std::numeric_limits<float>::epsilon();

        const Float3 hit_point = ray.point_at(distance);

        const Float3 center_to_hit = hit_point - center;
        const float u              = center_to_hit.dot(tangent);
        const float v              = center_to_hit.dot(bitangent);
        const float half_side      = side_length * 0.5f;

        if (std::abs(u) > half_side + epsilon || std::abs(v) > half_side + epsilon) {
            return hit;
        }

        hit.is_intersecting = true;
        hit.distance        = distance;
        hit.point           = hit_point;
        hit.normal          = unit_normal;

        return hit;
    }

    ATLAS_NODISCARD ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE HitSurface
    operator()(const Ray& ray) const noexcept {
        return trace(ray);
    }

private:
    ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE bool
    build_basis(const Float3& input_normal,
                Float3& unit_normal,
                Float3& tangent,
                Float3& bitangent) const noexcept {
        return atlas::orthonormal_basis(input_normal, unit_normal, tangent, bitangent);
    }
};

class Square::Builder final {
public:
    Builder() = default;

    ATLAS_NODISCARD ATLAS_HOST Square
    build() const;

    ATLAS_NODISCARD ATLAS_HOST atlas::host_shared_ptr<Square>
    make_host_shared() const;

    ATLAS_HOST Builder&
    with_center(const Float3& center_) noexcept;

    ATLAS_HOST Builder&
    with_normal(const Float3& normal_) noexcept;

    ATLAS_HOST Builder&
    with_side_length(float side_length_) noexcept;

private:
    ATLAS_HOST void
    validate() const;

private:
    Float3 _center     = Float3(0.0f, 0.0f, 0.0f);
    Float3 _normal     = Float3(0.0f, 0.0f, 1.0f);
    float _side_length = 1.0f;
};

using SquareHostPtr = atlas::host_shared_ptr<Square>;

using SquareDevicePtr = atlas::device_shared_ptr<Square>;

}
