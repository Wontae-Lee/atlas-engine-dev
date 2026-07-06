#pragma once

#include <atlas/core/macros.h>
#include <atlas/math/math.h>
#include <atlas/memory/memory.h>
#include <atlas/spatial/axis_aligned_bounding_box.h>
#include <atlas/spatial/ray.h>

#include <cmath>
#include <cstddef>
#include <optional>

namespace atlas {

class Triangle final {
public:
    class Builder;

public:
    Vector3 a = Vector3(0.0f, 0.0f, 0.0f);

    Vector3 b = Vector3(0.0f, 0.0f, 0.0f);

    Vector3 c = Vector3(0.0f, 0.0f, 0.0f);

    Vector3 n = Vector3(0.0f, 0.0f, 1.0f);

    Vector3 normal = Vector3(0.0f, 0.0f, 1.0f);

    Triangle() noexcept = default;

    ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE
    Triangle(const Vector3& a_, const Vector3& b_, const Vector3& c_) noexcept
        : a(a_)
        , b(b_)
        , c(c_) {
        normal = atlas::normalized_or(
            atlas::cross(b - a, c - a),
            Vector3(0.0f, 0.0f, 0.0f));
    }

    Triangle(const Triangle& other) noexcept            = default;
    Triangle(Triangle&& other) noexcept                 = default;
    Triangle& operator=(const Triangle& other) noexcept = default;
    Triangle& operator=(Triangle&& other) noexcept      = default;

    ~Triangle() noexcept = default;

    ATLAS_HOST ATLAS_NODISCARD static Builder
    builder() noexcept;

    ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE Vector3
    closest_point(const Vector3& p) const noexcept {
        const Vector3 v0 = a;
        const Vector3 v1 = b;
        const Vector3 v2 = c;

        const Vector3 ab = v1 - v0;
        const Vector3 ac = v2 - v0;
        const Vector3 ap = p - v0;

        const float d1 = ab.dot(ap);
        const float d2 = ac.dot(ap);

        if (d1 <= 0.0f && d2 <= 0.0f) {
            return v0;
        }

        const Vector3 bp = p - v1;
        const float d3   = ab.dot(bp);
        const float d4   = ac.dot(bp);

        if (d3 >= 0.0f && d4 <= d3) {
            return v1;
        }

        const float vc = d1 * d4 - d3 * d2;

        if (vc <= 0.0f && d1 >= 0.0f && d3 <= 0.0f) {
            const float vv = d1 / (d1 - d3);
            return v0 + ab * vv;
        }

        const Vector3 cpv = p - v2;
        const float d5    = ab.dot(cpv);
        const float d6    = ac.dot(cpv);

        if (d6 >= 0.0f && d5 <= d6) {
            return v2;
        }

        const float vb = d5 * d2 - d1 * d6;

        if (vb <= 0.0f && d2 >= 0.0f && d6 <= 0.0f) {
            const float ww = d2 / (d2 - d6);
            return v0 + ac * ww;
        }

        const float va = d3 * d6 - d5 * d4;

        if (va <= 0.0f && (d4 - d3) >= 0.0f && (d5 - d6) >= 0.0f) {
            const float ww = (d4 - d3) / ((d4 - d3) + (d5 - d6));
            return v1 + (v2 - v1) * ww;
        }

        const float denom = 1.0f / (va + vb + vc);
        const float vv    = vb * denom;
        const float ww    = vc * denom;

        return v0 + ab * vv + ac * ww;
    }

    ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE Vector3
    closest_normal(const Vector3&) const noexcept {
        return normal;
    }

    ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE float
    signed_distance(const Vector3& p) const noexcept {
        const Vector3 nn     = closest_normal(p);
        const float sd_plane = (p - a).dot(nn);

        const Vector3 cp = closest_point(p);
        const float d    = (p - cp).length();

        return (sd_plane >= 0.0f) ? d : -d;
    }

    ATLAS_ALL_DEVICE ATLAS_NODISCARD ATLAS_FORCE_INLINE bool
    is_inside(const Vector3& p, const float tolerance = 0.0f) const noexcept {
        const Vector3 nn = normal;

        const float nn_len2 = nn.length_squared();

        if (nn_len2 <= 0.0f) {
            return false;
        }

        const float side = (p - a).dot(nn);

        if (side <= 0.0f) {
            if (tolerance >= 0.0f) {
                return true;
            }
        } else if (tolerance < 0.0f) {
            return false;
        }

        const Vector3 cp = closest_point(p);
        const float d2   = (p - cp).length_squared();

        return side <= 0.0f ? d2 >= tolerance * tolerance
                            : d2 <= tolerance * tolerance;
    }

    ATLAS_ALL_DEVICE ATLAS_NODISCARD ATLAS_FORCE_INLINE bool
    is_on_surface(const Vector3& p, const float tolerance = 0.0f) const noexcept {
        if (tolerance < 0.0f) {
            return false;
        }

        const Vector3 cp = closest_point(p);
        return (p - cp).length_squared() <= tolerance * tolerance;
    }

    ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE Vector3
    centroid() const noexcept {
        return (a + b + c) * (1.0f / 3.0f);
    }

    ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE AABB
    bound() const noexcept {
        const Vector3 mn = atlas::cmin(a, atlas::cmin(b, c));
        const Vector3 mx = atlas::cmax(a, atlas::cmax(b, c));

        return AABB(mn, mx);
    }

    ATLAS_ALL_DEVICE ATLAS_NODISCARD ATLAS_FORCE_INLINE bool
    is_valid() const noexcept {
        const Vector3 nn = atlas::cross(b - a, c - a);
        return nn.length_squared() > 0.0f;
    }

    ATLAS_ALL_DEVICE ATLAS_NODISCARD ATLAS_FORCE_INLINE HitSurface
    trace(const Ray& r) const noexcept {
        HitSurface result {};

        const Vector3 v0 = a;
        const Vector3 v1 = b;
        const Vector3 v2 = c;

        const Vector3 e1 = v1 - v0;
        const Vector3 e2 = v2 - v0;

        const Vector3 pvec = atlas::cross(r.direction, e2);
        const float det    = e1.dot(pvec);

        if (std::abs(det) <= eps) {
            return result;
        }

        const float inv_det = 1.0f / det;

        const Vector3 tvec = r.origin - v0;
        const float u      = tvec.dot(pvec) * inv_det;

        if (u < 0.0f || u > 1.0f) {
            return result;
        }

        const Vector3 qvec = atlas::cross(tvec, e1);
        const float v      = r.direction.dot(qvec) * inv_det;

        if (v < 0.0f || (u + v) > 1.0f) {
            return result;
        }

        const float t = e2.dot(qvec) * inv_det;

        if (t < eps) {
            return result;
        }

        result.is_intersecting = true;
        result.distance        = t;
        result.point           = r.point_at(t);

        const Vector3 normal_vec = normal;

        result.normal = atlas::normalized_or(
            normal_vec,
            Vector3(1.0f, 0.0f, 0.0f));

        return result;
    }

    ATLAS_ALL_DEVICE ATLAS_NODISCARD ATLAS_FORCE_INLINE HitSurface
    operator()(const Ray& ray) const noexcept {
        return trace(ray);
    }
};

class Triangle::Builder final {
public:
    Builder() = default;

    ATLAS_HOST ATLAS_NODISCARD Triangle
    build() const;

    ATLAS_HOST ATLAS_NODISCARD atlas::host_shared_ptr<Triangle>
    make_host_shared() const;

    ATLAS_HOST Builder&
    with_a(const Vector3& a_) noexcept;

    ATLAS_HOST Builder&
    with_b(const Vector3& b_) noexcept;

    ATLAS_HOST Builder&
    with_c(const Vector3& c_) noexcept;

    ATLAS_HOST Builder&
    with_vertices(const Vector3& a_, const Vector3& b_, const Vector3& c_) noexcept;

    ATLAS_HOST Builder&
    with_normal(const Vector3& normal_) noexcept;

private:
    ATLAS_HOST void
    validate() const;

private:
    Vector3 _a = Vector3(0.0f, 0.0f, 0.0f);

    Vector3 _b = Vector3(0.0f, 0.0f, 0.0f);

    Vector3 _c = Vector3(0.0f, 0.0f, 0.0f);

    std::optional<Vector3> _normal;
};

using TriangleHostPtr = atlas::host_shared_ptr<Triangle>;

using TriangleDevicePtr = atlas::device_shared_ptr<Triangle>;

}
