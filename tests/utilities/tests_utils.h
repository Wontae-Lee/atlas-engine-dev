#pragma once

#include <atlas/atlas.h>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <type_traits>
#include <utility>

namespace atlas::test {

template <typename T>
static ATLAS_FORCE_INLINE bool
near(T a, T b, T eps) {
    if constexpr (std::is_floating_point_v<T>) {
        return std::abs(a - b) <= eps;
    } else {
        return a == b;
    }
}

template <typename V>
static ATLAS_FORCE_INLINE bool
is_finite_vec(const V& v) {
    using T = decltype(v[0]);
    if constexpr (std::is_floating_point_v<T>) {
        for (std::size_t i = 0; i < V::size(); ++i) {
            if (!std::isfinite(static_cast<double>(v[i]))) return false;
        }
    }
    return true;
}

template <typename T, std::size_t N>
static ATLAS_FORCE_INLINE bool
vec_near(const atlas::Vector<T, N>& a,
         const atlas::Vector<T, N>& b,
         T eps) {
    for (std::size_t i = 0; i < N; ++i) {
        if (!near<T>(a[i], b[i], eps)) return false;
    }
    return true;
}

template <typename T>
static ATLAS_FORCE_INLINE bool
vec_near(const atlas::Vector<T, 3>& a,
         const atlas::Vector<T, 3>& b,
         T eps) {
    for (std::size_t i = 0; i < 3; ++i) {
        if (!near<T>(a[i], b[i], eps)) return false;
    }
    return true;
}

template <typename Container, typename T, std::size_t N>
static ATLAS_FORCE_INLINE bool
contains_point(const Container& points,
               const atlas::Vector<T, N>& expected,
               T eps) {
    return std::any_of(points.begin(), points.end(), [&](const auto& point) {
        return vec_near(point, expected, eps);
    });
}

struct Foo {
    int x    = 0;
    double y = 0.0;

    Foo() = default;

    Foo(const int x_, const double y_)
        : x(x_)
        , y(y_) { }
};

template <typename T>
class DummyCodec final : public system::Codec<T> {
public:
    using Base = system::Codec<T>;

    bool encode_called = false;
    bool decode_called = false;

public:
    DummyCodec() = default;

    explicit DummyCodec(DomainHostPtr<T> domain)
        : Base(domain) { }

    void
    encode(const system::ParticleDeviceProbe<T>&,
           const system::DomainDeviceProbe<T>&,
           const system::SpatialHashingProbe<T>&,
           system::CodecDeviceProbe<T>&) override {
        encode_called = true;
    }

    void
    decode(const system::ParticleDeviceProbe<T>&,
           const system::DomainDeviceProbe<T>&,
           const system::SpatialHashingProbe<T>&,
           system::CodecDeviceProbe<T>&) override {
        decode_called = true;
    }

    ATLAS_NODISCARD system::CodecType
    type() const noexcept override {
        return system::CodecType::single;
    }
};

template <typename T>
system::SyncOperator<T>
make_sync_operator(const Vector3<T>& translation,
                   const math::Quaternion<T>& orientation) {
    system::SyncOperator<T> op(translation, orientation);
    op.rebuild_matrices();
    return op;
}

template <typename T>
atlas::SyncHostPtr<T>
make_host_shared_sync() {
    const Vector3<T> translation(T(0), T(0), T(0));
    const math::Quaternion<T> orientation;

    return atlas::make_host_shared<system::Sync<T>>(
        translation,
        orientation);
}

template <typename T>
atlas::SyncHostPtr<T>
make_host_shared_sync(const Vector3<T>& translation,
                      const math::Quaternion<T>& orientation) {
    return atlas::make_host_shared<system::Sync<T>>(
        translation,
        orientation);
}

ATLAS_FORCE_INLINE geometry::Sphere<double>
make_sphere() {
    return { Vector3<double>(0.0, 0.0, 0.0), 1.0 };
}

ATLAS_FORCE_INLINE geometry::Box<double>
make_box() {
    return {
        Vector3<double>(-1.0, -1.0, -1.0),
        Vector3<double>(1.0, 1.0, 1.0)
    };
}

ATLAS_FORCE_INLINE geometry::Cylinder<double>
make_cylinder() {
    return {
        Vector3<double>(0.0, 0.0, 0.0),
        1.0,
        2.0
    };
}

ATLAS_FORCE_INLINE geometry::Triangle<double>
make_triangle() {
    return {
        Vector3<double>(0.0, 0.0, 0.0),
        Vector3<double>(1.0, 0.0, 0.0),
        Vector3<double>(0.0, 1.0, 0.0)
    };
}

template <typename T>
ATLAS_FORCE_INLINE geometry::QueryOperator<T>
make_box_query_operator(const geometry::Box<T>& box) {
    return box.make_query_operator();
}

template <typename T>
ATLAS_FORCE_INLINE geometry::QueryOperator<T>
make_sphere_query_operator(const geometry::Sphere<T>& sphere) {
    return sphere.make_query_operator();
}

template <typename T>
ATLAS_FORCE_INLINE geometry::QueryOperator<T>
make_cylinder_query_operator(const geometry::Cylinder<T>& cylinder) {
    return cylinder.make_query_operator();
}

template <typename T>
ATLAS_FORCE_INLINE geometry::QueryOperator<T>
make_triangle_query_operator(const geometry::Triangle<T>& triangle) {
    return triangle.make_query_operator();
}

ATLAS_FORCE_INLINE host_shared_ptr<geometry::Sphere<double>>
make_host_shared_sphere() {
    return atlas::make_host_shared<geometry::Sphere<double>>(make_sphere());
}

ATLAS_FORCE_INLINE system::Domain<double>
make_domain() {
    const Vector3<double> lower(-1.0, -1.0, -1.0);
    const Vector3<double> upper(1.0, 1.0, 1.0);
    constexpr double h = 0.5;
    return { lower, upper, h };
}

ATLAS_FORCE_INLINE host_shared_ptr<system::Domain<double>>
make_domain_ptr() {
    return atlas::make_host_shared<system::Domain<double>>(make_domain());
}

} // namespace atlas::test
