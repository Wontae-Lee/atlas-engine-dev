#pragma once

#include <atlas/atlas.h>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <type_traits>
#include <utility>
#include <vector>

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

template <typename V, typename T>
static ATLAS_FORCE_INLINE bool
vec_length_near(const V& v, T expected, T eps) {
    return near<T>(v.length(), expected, eps);
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

template <typename Container>
static ATLAS_FORCE_INLINE bool
all_finite_points(const Container& points) {
    return std::all_of(points.begin(), points.end(), [](const auto& point) {
        return is_finite_vec(point);
    });
}

template <typename Container, typename T>
static ATLAS_FORCE_INLINE bool
points_in_range(const Container& points, T min_value, T max_value) {
    return std::all_of(points.begin(), points.end(), [&](const auto& point) {
        for (std::size_t i = 0; i < 3; ++i) {
            if (point[i] < min_value || point[i] > max_value) return false;
        }
        return true;
    });
}

template <typename ContainerA, typename ContainerB, typename T>
static ATLAS_FORCE_INLINE bool
point_buffers_near(const ContainerA& a, const ContainerB& b, T eps) {
    if (a.size() != b.size()) return false;

    for (std::size_t i = 0; i < a.size(); ++i) {
        if (!vec_near(a[i], b[i], eps)) return false;
    }
    return true;
}

template <typename T>
static ATLAS_FORCE_INLINE std::vector<T>
copy_device_range(const T* src, std::size_t count) {
    std::vector<T> host(count);
    if (count > 0) {
        atlas::copy_device_to_host(src, host.data(), count);
    }
    return host;
}

template <typename T>
static ATLAS_FORCE_INLINE std::vector<T>
copy_device_buffer(const atlas::DeviceBuffer<T>& src) {
    return copy_device_range(atlas::raw_pointer_cast(src.data()), src.size());
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
    encode(const system::FluidDeviceProbe<T>&,
           const system::DomainDeviceProbe<T>&,
           const system::SpatialHashingProbe<T>&,
           system::CodecDeviceProbe<T>&) override {
        encode_called = true;
    }

    void
    decode(const system::FluidDeviceProbe<T>&,
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
class DummyMeasure final : public system::Measure<T> {
public:
    void
    measure(system::DomainDeviceProbe<T>, system::SpatialHashingProbe<T>, system::FluidDeviceProbe<T>) override {
        ++call_count;
    }

    int call_count = 0;
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
ATLAS_FORCE_INLINE geometry::GeometryOperator<T>
make_box_geometry_operator(const geometry::Box<T>& box) {
    return box.make_geometry_operator();
}

template <typename T>
ATLAS_FORCE_INLINE geometry::GeometryOperator<T>
make_sphere_geometry_operator(const geometry::Sphere<T>& sphere) {
    return sphere.make_geometry_operator();
}

template <typename T>
ATLAS_FORCE_INLINE geometry::GeometryOperator<T>
make_cylinder_geometry_operator(const geometry::Cylinder<T>& cylinder) {
    return cylinder.make_geometry_operator();
}

template <typename T>
ATLAS_FORCE_INLINE geometry::GeometryOperator<T>
make_triangle_geometry_operator(const geometry::Triangle<T>& triangle) {
    return triangle.make_geometry_operator();
}

ATLAS_FORCE_INLINE host_shared_ptr<geometry::Sphere<double>>
make_host_shared_sphere() {
    return atlas::make_host_shared<geometry::Sphere<double>>(make_sphere());
}

template <typename T>
ATLAS_FORCE_INLINE atlas::UnitHostPtr<T>
make_host_shared_unit() {
    auto geometry = atlas::make_host_shared<geometry::Sphere<T>>(
        geometry::Sphere<T>(Vector3<T>(T(0), T(0), T(0)), T(2)));
    auto sync = atlas::make_host_shared<system::Sync<T>>();

    return system::Unit<T>::builder()
        .with_geometry(geometry)
        .with_sync(sync)
        .make_host_shared();
}

template <typename T>
ATLAS_FORCE_INLINE atlas::ColliderSurfaceInteractionHostPtr<T>
make_host_shared_collider_surface_interaction() {
    return atlas::ColliderSurfaceInteraction<T>::builder()
        .with_diffuse_sampling(system::DiffuseSampling::CosineWeighted)
        .with_restitution(T(0.7))
        .with_tangential_momentum_accommodation(T(0.3))
        .with_temperature(T(325))
        .make_host_shared();
}

template <typename T>
ATLAS_FORCE_INLINE atlas::ColliderHostPtr<T>
make_host_shared_collider() {
    return atlas::Collider<T>::builder()
        .with_unit(make_host_shared_unit<T>())
        .with_surface_interaction(make_host_shared_collider_surface_interaction<T>())
        .make_host_shared();
}

template <typename T>
ATLAS_FORCE_INLINE system::Domain<T>
make_domain() {
    const Vector3<T> lower(T(-1), T(-1), T(-1));
    const Vector3<T> upper(T(1), T(1), T(1));
    const T h = T(0.5);
    return { lower, upper, h };
}

ATLAS_FORCE_INLINE system::Domain<double>
make_domain() {
    return make_domain<double>();
}

template <typename T>
ATLAS_FORCE_INLINE host_shared_ptr<system::Domain<T>>
make_domain_ptr() {
    return atlas::make_host_shared<system::Domain<T>>(make_domain<T>());
}

template <typename T>
ATLAS_FORCE_INLINE host_shared_ptr<system::Domain<T>>
make_isothermal_domain_ptr(T temperature = T(300)) {
    return system::Domain<T>::builder()
        .with_lower_corner(Vector3<T>(T(0), T(0), T(0)))
        .with_upper_corner(Vector3<T>(T(1), T(1), T(1)))
        .with_cell_size(T(0.5))
        .with_type(system::DomainType::isothermal)
        .with_temperature(temperature)
        .make_host_shared();
}

template <typename T>
ATLAS_FORCE_INLINE system::SpatialHashingSearcher<T>
make_single_range_searcher(const atlas::DomainHostPtr<T>& domain) {
    return atlas::SpatialHashingSearcher<T>::builder()
        .with_domain(domain)
        .with_range(system::NeighborSearchRange::single)
        .build();
}

ATLAS_FORCE_INLINE host_shared_ptr<system::Domain<double>>
make_domain_ptr() {
    return make_domain_ptr<double>();
}

} // namespace atlas::test

#ifdef ATLAS_ENABLE_VIZKIT

#include <vizkit/layer/geometry/geometry_layer.h>
#include <vizkit/layer/layer.h>

namespace atlas::test {

template <typename T>
atlas::UnitHostPtr<T>
make_vizkit_translated_box_unit(const atlas::Vector3<T>& translation) {
    const auto geometry = atlas::geometry::Box<T>::builder()
                              .with_lower_corner(atlas::Vector3<T>(T(-1), T(-1), T(-1)))
                              .with_upper_corner(atlas::Vector3<T>(T(1), T(1), T(1)))
                              .make_host_shared();
    const auto sync = atlas::system::Sync<T>::builder()
                          .with_rigid_pose(translation, atlas::Quaternion<T>())
                          .make_host_shared();
    return atlas::system::Unit<T>::builder()
        .with_geometry(geometry)
        .with_sync(sync)
        .make_host_shared();
}

template <typename T>
atlas::UnitHostPtr<T>
make_vizkit_box_unit(const atlas::Vector3<T>& lower,
                     const atlas::Vector3<T>& upper) {
    const auto geometry = atlas::geometry::Box<T>::builder()
                              .with_lower_corner(lower)
                              .with_upper_corner(upper)
                              .make_host_shared();
    const auto sync = atlas::system::Sync<T>::builder().make_host_shared();
    return atlas::system::Unit<T>::builder()
        .with_geometry(geometry)
        .with_sync(sync)
        .make_host_shared();
}

template <typename T>
class TestVizkitGeometryLayer final : public atlas::vizkit::GeometryLayer<T> {
public:
    explicit TestVizkitGeometryLayer(const atlas::UnitHostPtr<T>& unit = nullptr)
        : atlas::vizkit::GeometryLayer<T>(GL_LINES, unit) { }

    void
    build_geometry(std::vector<atlas::Vector3<T>>& positions) override {
        positions = seeded_positions;
    }

    bool
    synchronize_public(std::vector<atlas::Vector3<T>>& world_positions, T dt) {
        return this->synchronize(world_positions, dt);
    }

    std::vector<atlas::Vector3<T>>&
    local_positions_public() {
        return this->_local_positions;
    }

    std::vector<atlas::Vector3<T>> seeded_positions;
};

template <typename T>
class DummyVizkitLayer final : public atlas::vizkit::Layer<T> {
public:
    void
    init(GLFWwindow* window, atlas::vizkit::Camera& camera) override {
        init_called = true;
        last_window = window;
        (void)camera;
    }

    void
    update(GLFWwindow* window, atlas::vizkit::Camera& camera, T dt) override {
        update_called = true;
        last_window   = window;
        last_dt       = dt;
        (void)camera;
    }

    bool init_called = false;
    bool update_called = false;
    GLFWwindow* last_window = nullptr;
    T last_dt = T(0);
};

} // namespace atlas::test

#endif
