#pragma once

/**
 * @file test_utils.h
 * @brief Declares reusable test helpers, dummy runtime objects, and optional Vizkit test fixtures.
 *
 * @details
 * This header provides a collection of small utilities used across Atlas test code.
 * It includes:
 * - scalar and vector comparison helpers,
 * - finite-value validation helpers,
 * - host-side copies of device ranges and buffers,
 * - dummy codec and measure implementations for runtime orchestration tests,
 * - convenience factories for common geometry, domain, sync, fluid, source, sink,
 *   collider, and unit test objects,
 * - optional Vizkit-specific test fixtures when `ATLAS_ENABLE_VIZKIT` is enabled.
 *
 * ## Purpose
 * These utilities reduce boilerplate in tests by centralizing common patterns such as:
 * - approximate floating-point comparison,
 * - copying device buffers to host vectors,
 * - constructing small canonical geometry objects,
 * - creating fully configured runtime objects suitable for integration tests,
 * - providing simple mock-like implementations of abstract interfaces.
 *
 * ## Design notes
 * - Most helpers are intentionally lightweight and header-only.
 * - Many factories return fully initialized objects with stable, deterministic defaults.
 * - Approximate comparisons are designed primarily for floating-point tests.
 * - Dummy runtime classes record whether specific methods were called, making them
 *   useful for orchestration and pipeline verification.
 *
 * ## Conditional Vizkit support
 * When `ATLAS_ENABLE_VIZKIT` is defined, this header also exposes:
 * - unit factories tailored for visualization tests,
 * - a dummy Vizkit layer,
 * - a test geometry layer exposing otherwise protected synchronization helpers.
 *
 * ---
 */

#include <atlas/atlas.h>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <type_traits>
#include <vector>

namespace atlas::test {

/**
 * @brief Compare two scalar values for approximate equality.
 *
 * @details
 * For floating-point types, the comparison uses an absolute tolerance:
 * \f[
 * |a - b| \le \epsilon
 * \f]
 *
 * For non-floating-point types, the comparison falls back to exact equality.
 *
 * @param a First value.
 * @param b Second value.
 * @param eps Allowed absolute tolerance for floating-point comparisons.
 * @return `true` if the values are considered equal; otherwise `false`.
 *
 * @tparam T Scalar type.
 */
template <typename T>
static ATLAS_FORCE_INLINE bool
near(T a, T b, T eps) {
    if constexpr (std::is_floating_point_v<T>) {
        return std::abs(a - b) <= eps;
    } else {
        return a == b;
    }
}

/**
 * @brief Return whether every component of a vector-like object is finite.
 *
 * @details
 * For floating-point component types, each element is checked with `std::isfinite`.
 * For non-floating-point component types, the function returns `true`.
 *
 * The vector-like type is expected to provide:
 * - `operator[]`,
 * - a static `size()` member.
 *
 * @param v Vector-like object to validate.
 * @return `true` if all floating-point components are finite; otherwise `false`.
 *
 * @tparam V Vector-like type.
 */
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

/**
 * @brief Compare two Atlas vectors component-wise using an absolute tolerance.
 *
 * @param a First vector.
 * @param b Second vector.
 * @param eps Allowed per-component absolute tolerance.
 * @return `true` if all components are approximately equal; otherwise `false`.
 *
 * @tparam T Scalar type.
 * @tparam N Vector dimension.
 */
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

/**
 * @brief Compare a vector length against an expected value using a tolerance.
 *
 * @param v Vector-like object supporting `length()`.
 * @param expected Expected vector magnitude.
 * @param eps Allowed absolute tolerance.
 * @return `true` if the vector length is approximately equal to @p expected.
 *
 * @tparam V Vector-like type.
 * @tparam T Scalar type.
 */
template <typename V, typename T>
static ATLAS_FORCE_INLINE bool
vec_length_near(const V& v, T expected, T eps) {
    return near<T>(v.length(), expected, eps);
}

/**
 * @brief Return whether every point in a container has only finite components.
 *
 * @param points Container of vector-like points.
 * @return `true` if all points are finite; otherwise `false`.
 *
 * @tparam Container Container type whose elements are vector-like objects.
 */
template <typename Container>
static ATLAS_FORCE_INLINE bool
all_finite_points(const Container& points) {
    return std::all_of(points.begin(), points.end(), [](const auto& point) {
        return is_finite_vec(point);
    });
}

/**
 * @brief Return whether a container contains a point approximately equal to an expected point.
 *
 * @param points Container of points to search.
 * @param expected Expected point.
 * @param eps Allowed per-component tolerance.
 * @return `true` if any point in the container is approximately equal to @p expected.
 *
 * @tparam Container Container type.
 * @tparam T Scalar type.
 * @tparam N Point dimension.
 */
template <typename Container, typename T, std::size_t N>
static ATLAS_FORCE_INLINE bool
contains_point(const Container& points,
               const atlas::Vector<T, N>& expected,
               T eps) {
    return std::any_of(points.begin(), points.end(), [&](const auto& point) {
        return vec_near(point, expected, eps);
    });
}

/**
 * @brief Compare two point buffers element-wise using approximate vector equality.
 *
 * @param a First point container.
 * @param b Second point container.
 * @param eps Allowed per-component tolerance.
 * @return `true` if both buffers have equal size and each corresponding point is approximately equal.
 *
 * @tparam ContainerA First container type.
 * @tparam ContainerB Second container type.
 * @tparam T Scalar type.
 */
template <typename ContainerA, typename ContainerB, typename T>
static ATLAS_FORCE_INLINE bool
point_buffers_near(const ContainerA& a, const ContainerB& b, T eps) {
    if (a.size() != b.size()) return false;

    for (std::size_t i = 0; i < a.size(); ++i) {
        if (!vec_near(a[i], b[i], eps)) return false;
    }

    return true;
}

/**
 * @brief Return whether all point coordinates lie within a closed scalar range.
 *
 * @details
 * This helper assumes each point has at least three Cartesian components and
 * checks each coordinate against:
 * \f[
 * \texttt{min\_value} \le p_i \le \texttt{max\_value}
 * \f]
 *
 * @param points Container of 3D points.
 * @param min_value Minimum allowed coordinate value.
 * @param max_value Maximum allowed coordinate value.
 * @return `true` if all points lie inside the scalar range for all checked coordinates.
 *
 * @tparam Container Container type.
 * @tparam T Scalar type.
 */
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

/**
 * @brief Copy a raw device range into a host-side `std::vector`.
 *
 * @details
 * The function allocates a host vector of size @p count and copies the contents
 * of the device memory range beginning at @p src into it using
 * `atlas::copy_device_to_host`.
 *
 * @param src Pointer to the source device range.
 * @param count Number of elements to copy.
 * @return Host vector containing a copy of the device data.
 *
 * @tparam T Element type.
 */
template <typename T>
static ATLAS_FORCE_INLINE std::vector<T>
copy_device_range(const T* src, std::size_t count) {
    std::vector<T> host(count);

    if (count > 0) {
        atlas::copy_device_to_host(src, host.data(), count);
    }

    return host;
}

/**
 * @brief Copy an Atlas device buffer into a host-side `std::vector`.
 *
 * @param src Source device buffer.
 * @return Host vector containing the copied buffer contents.
 *
 * @tparam T Element type.
 */
template <typename T>
static ATLAS_FORCE_INLINE std::vector<T>
copy_device_buffer(const atlas::DeviceBuffer<T>& src) {
    return copy_device_range(atlas::raw_pointer_cast(src.data()), src.size());
}

/**
 * @brief Small aggregate type used in generic utility and memory tests.
 *
 * @details
 * This type is intentionally simple and trivially understandable, making it
 * useful for tests involving:
 * - buffer transfers,
 * - object construction,
 * - copy/move behavior,
 * - generic container handling.
 */
struct Foo {
    /**
     * @brief Integer payload field.
     */
    int x = 0;

    /**
     * @brief Floating-point payload field.
     */
    double y = 0.0;

    /**
     * @brief Default constructor.
     */
    Foo() = default;

    /**
     * @brief Construct from explicit field values.
     *
     * @param x_ Integer payload.
     * @param y_ Floating-point payload.
     */
    Foo(const int x_, const double y_)
        : x(x_)
        , y(y_) { }
};

/**
 * @brief Dummy codec used for runtime orchestration tests.
 *
 * @details
 * This test double derives from @ref system::Codec and records whether
 * @ref encode or @ref decode were invoked.
 *
 * It is useful for verifying:
 * - system update ordering,
 * - codec-stage integration,
 * - pipeline execution paths.
 *
 * @tparam T Floating-point scalar type.
 */
template <typename T>
class DummyCodec final : public system::Codec<T> {
public:
    /**
     * @brief Base codec type alias.
     */
    using Base = system::Codec<T>;

    /**
     * @brief Whether @ref encode has been called.
     */
    bool encode_called = false;

    /**
     * @brief Whether @ref decode has been called.
     */
    bool decode_called = false;

public:
    /**
     * @brief Default constructor.
     */
    DummyCodec() = default;

    /**
     * @brief Construct the dummy codec with an associated domain.
     *
     * @param domain Domain supplied to the base codec.
     */
    explicit DummyCodec(DomainHostPtr<T> domain)
        : Base(domain) { }

    /**
     * @brief Record that encoding was requested.
     */
    void
    encode(const system::FluidDeviceProbe<T>&,
           const system::Universe<T>&,
           const system::SpatialHashingProbe<T>&,
           system::CodecDeviceProbe<T>&) override {
        encode_called = true;
    }

    /**
     * @brief Record that decoding was requested.
     */
    void
    decode(const system::FluidDeviceProbe<T>&,
           const system::Universe<T>&,
           const system::SpatialHashingProbe<T>&,
           system::CodecDeviceProbe<T>&) override {
        decode_called = true;
    }

    /**
     * @brief Return a dummy runtime codec tag.
     *
     * @return `system::CodecType::single`.
     */
    ATLAS_NODISCARD system::CodecType
    type() const noexcept override {
        return system::CodecType::single;
    }
};

/**
 * @brief Dummy measure used for orchestration and callback-count tests.
 *
 * @details
 * This test double derives from @ref system::Measure and increments
 * @ref call_count each time @ref measure is invoked.
 *
 * @tparam T Floating-point scalar type.
 */
template <typename T>
class DummyMeasure final : public system::Measure<T> {
public:
    /**
     * @brief Number of times @ref measure has been called.
     */
    int call_count = 0;

public:
    /**
     * @brief Increment the invocation counter.
     */
    void
    measure(system::Universe<T>&, system::SpatialHashingProbe<T>, system::FluidDeviceProbe<T>) override {
        ++call_count;
    }

    /**
     * @brief Return the measurement mode supported by this dummy measure.
     *
     * @return `system::MeasureModeType::All`.
     */
    ATLAS_ALL_DEVICE ATLAS_NODISCARD system::MeasureModeType
    measure_mode() const noexcept override {
        return system::MeasureModeType::All;
    }
};

/**
 * @brief Return a canonical test box centered at the origin.
 *
 * @return Box with corners `(-1,-1,-1)` and `(1,1,1)`.
 */
ATLAS_FORCE_INLINE geometry::Box<double>
make_box() {
    return {
        Vector3<double>(-1.0, -1.0, -1.0),
        Vector3<double>(1.0, 1.0, 1.0)
    };
}

/**
 * @brief Return a canonical test cylinder centered at the origin.
 *
 * @return Cylinder with radius `1.0` and height `2.0`.
 */
ATLAS_FORCE_INLINE geometry::Cylinder<double>
make_cylinder() {
    return {
        Vector3<double>(0.0, 0.0, 0.0),
        1.0,
        2.0
    };
}

/**
 * @brief Return a canonical test sphere centered at the origin.
 *
 * @return Sphere with radius `1.0`.
 */
ATLAS_FORCE_INLINE geometry::Sphere<double>
make_sphere() {
    return { Vector3<double>(0.0, 0.0, 0.0), 1.0 };
}

/**
 * @brief Return a canonical right triangle in the `z = 0` plane.
 *
 * @return Triangle with vertices `(0,0,0)`, `(1,0,0)`, and `(0,1,0)`.
 */
ATLAS_FORCE_INLINE geometry::Triangle<double>
make_triangle() {
    return {
        Vector3<double>(0.0, 0.0, 0.0),
        Vector3<double>(1.0, 0.0, 0.0),
        Vector3<double>(0.0, 1.0, 0.0)
    };
}

/**
 * @brief Return a geometry operator bound to a box.
 *
 * @param box Source box.
 * @return Geometry operator exported from @p box.
 *
 * @tparam T Floating-point scalar type.
 */
template <typename T>
ATLAS_FORCE_INLINE geometry::GeometryOperator<T>
make_box_geometry_operator(const geometry::Box<T>& box) {
    return box.make_geometry_operator();
}

/**
 * @brief Return a geometry operator bound to a cylinder.
 *
 * @param cylinder Source cylinder.
 * @return Geometry operator exported from @p cylinder.
 *
 * @tparam T Floating-point scalar type.
 */
template <typename T>
ATLAS_FORCE_INLINE geometry::GeometryOperator<T>
make_cylinder_geometry_operator(const geometry::Cylinder<T>& cylinder) {
    return cylinder.make_geometry_operator();
}

/**
 * @brief Return a geometry operator bound to a sphere.
 *
 * @param sphere Source sphere.
 * @return Geometry operator exported from @p sphere.
 *
 * @tparam T Floating-point scalar type.
 */
template <typename T>
ATLAS_FORCE_INLINE geometry::GeometryOperator<T>
make_sphere_geometry_operator(const geometry::Sphere<T>& sphere) {
    return sphere.make_geometry_operator();
}

/**
 * @brief Return a geometry operator bound to a triangle.
 *
 * @param triangle Source triangle.
 * @return Geometry operator exported from @p triangle.
 *
 * @tparam T Floating-point scalar type.
 */
template <typename T>
ATLAS_FORCE_INLINE geometry::GeometryOperator<T>
make_triangle_geometry_operator(const geometry::Triangle<T>& triangle) {
    return triangle.make_geometry_operator();
}

/**
 * @brief Create a host-shared canonical sphere.
 *
 * @return Host-owned shared pointer to a sphere centered at the origin with radius `1.0`.
 */
ATLAS_FORCE_INLINE atlas::host_shared_ptr<geometry::Sphere<double>>
make_host_shared_sphere() {
    return atlas::make_host_shared<geometry::Sphere<double>>(make_sphere());
}

/**
 * @brief Create a sync operator from translation and orientation.
 *
 * @details
 * The operator is constructed and then explicitly asked to rebuild its internal
 * transformation matrices.
 *
 * @param translation World-space translation.
 * @param orientation Orientation quaternion.
 * @return Initialized sync operator.
 *
 * @tparam T Floating-point scalar type.
 */
template <typename T>
system::SyncOperator<T>
make_sync_operator(const Vector3<T>& translation,
                   const math::Quaternion<T>& orientation) {
    system::SyncOperator<T> op(translation, orientation);
    op.rebuild_matrices();
    return op;
}

/**
 * @brief Create a host-shared sync object at the identity pose.
 *
 * @return Host-owned shared pointer to a sync object at zero translation and identity orientation.
 *
 * @tparam T Floating-point scalar type.
 */
template <typename T>
atlas::SyncHostPtr<T>
make_host_shared_sync() {
    return make_host_shared_sync(Vector3<T>(T(0), T(0), T(0)), math::Quaternion<T> {});
}

/**
 * @brief Create a host-shared sync object from translation and orientation.
 *
 * @param translation World-space translation.
 * @param orientation Orientation quaternion.
 * @return Host-owned shared pointer to the constructed sync object.
 *
 * @tparam T Floating-point scalar type.
 */
template <typename T>
atlas::SyncHostPtr<T>
make_host_shared_sync(const Vector3<T>& translation,
                      const math::Quaternion<T>& orientation) {
    return atlas::make_host_shared<system::Sync<T>>(translation, orientation);
}

/**
 * @brief Create a box unit translated to a given world-space position.
 *
 * @details
 * The unit uses:
 * - a box spanning `[-1,+1]^3` in local space,
 * - a rigid pose placing it at @p translation,
 * - no explicit dynamic motion state.
 *
 * @param translation World-space translation of the unit.
 * @return Constructed box unit.
 *
 * @tparam T Floating-point scalar type.
 */
template <typename T>
ATLAS_FORCE_INLINE atlas::Unit<T>
make_box_unit(const atlas::Vector3<T>& translation = atlas::Vector3<T>(T(0), T(0), T(0))) {
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
        .build();
}

/**
 * @brief Create a host-shared sphere unit with default sync.
 *
 * @return Host-owned shared pointer to a unit whose geometry is a sphere of radius `2`.
 *
 * @tparam T Floating-point scalar type.
 */
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

/**
 * @brief Create a host-shared collider surface interaction with deterministic defaults.
 *
 * @return Host-owned shared pointer to a configured collider surface interaction.
 *
 * @tparam T Floating-point scalar type.
 */
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

/**
 * @brief Create a host-shared collider containing one default unit and one interaction.
 *
 * @return Host-owned shared pointer to a collider suitable for simple tests.
 *
 * @tparam T Floating-point scalar type.
 */
template <typename T>
ATLAS_FORCE_INLINE atlas::ColliderHostPtr<T>
make_host_shared_collider() {
    HostBuffer<system::Unit<T>> units { *make_host_shared_unit<T>() };
    HostBuffer<system::ColliderSurfaceInteraction<T>> interactions {
        *make_host_shared_collider_surface_interaction<T>()
    };

    return atlas::Collider<T>::builder()
        .with_units(units)
        .with_surface_interactions(interactions)
        .make_host_shared();
}

/**
 * @brief Create a simple cubic domain centered at the origin.
 *
 * @details
 * The returned domain spans:
 * - lower corner `(-1,-1,-1)`
 * - upper corner `(1,1,1)`
 * - cell size `0.5`
 *
 * @return Constructed domain value.
 *
 * @tparam T Floating-point scalar type.
 */
template <typename T>
ATLAS_FORCE_INLINE system::Domain<T>
make_domain() {
    const Vector3<T> lower(T(-1), T(-1), T(-1));
    const Vector3<T> upper(T(1), T(1), T(1));
    const T h = T(0.5);

    return { lower, upper, h };
}

/**
 * @brief Return the default double-precision test domain.
 *
 * @return Constructed double-precision domain.
 */
ATLAS_FORCE_INLINE system::Domain<double>
make_domain() {
    return make_domain<double>();
}

/**
 * @brief Create a host-shared pointer to the default test domain.
 *
 * @return Host-owned shared pointer to the default domain.
 *
 * @tparam T Floating-point scalar type.
 */
template <typename T>
ATLAS_FORCE_INLINE atlas::host_shared_ptr<system::Domain<T>>
make_domain_ptr() {
    return atlas::make_host_shared<system::Domain<T>>(make_domain<T>());
}

/**
 * @brief Return the default double-precision host-shared test domain.
 *
 * @return Host-owned shared pointer to the default double-precision domain.
 */
ATLAS_FORCE_INLINE atlas::host_shared_ptr<system::Domain<double>>
make_domain_ptr() {
    return make_domain_ptr<double>();
}

/**
 * @brief Create a host-shared isothermal domain with a configurable temperature.
 *
 * @param temperature Prescribed isothermal field temperature.
 * @return Host-owned shared pointer to the configured isothermal domain.
 *
 * @tparam T Floating-point scalar type.
 */
template <typename T>
ATLAS_FORCE_INLINE atlas::host_shared_ptr<system::Domain<T>>
make_isothermal_domain_ptr(T temperature = T(300)) {
    return system::Domain<T>::builder()
        .with_lower_corner(Vector3<T>(T(0), T(0), T(0)))
        .with_upper_corner(Vector3<T>(T(1), T(1), T(1)))
        .with_cell_size(T(0.5))
        .with_type(system::DomainType::isothermal)
        .with_temperature(temperature)
        .make_host_shared();
}

/**
 * @brief Create a spatial hashing searcher bound to a domain.
 *
 * @param domain Associated domain.
 * @return Configured spatial hashing searcher value.
 *
 * @tparam T Floating-point scalar type.
 */
template <typename T>
ATLAS_FORCE_INLINE system::SpatialHashingSearcher<T>
make_single_range_searcher(const atlas::DomainHostPtr<T>& domain) {
    return atlas::SpatialHashingSearcher<T>::builder()
        .with_domain(domain)
        .build();
}

/**
 * @brief Create a buffered fluid with the given particle capacity.
 *
 * @param buffer_size Total particle buffer capacity.
 * @return Host-owned shared pointer to the constructed fluid.
 *
 * @tparam T Floating-point scalar type.
 */
template <typename T>
atlas::FluidHostPtr<T>
make_buffered_fluid(const std::size_t buffer_size) {
    return atlas::system::Fluid<T>::builder()
        .with_buffer_size(buffer_size)
        .make_host_shared();
}

/**
 * @brief Create a small two-species fluid suitable for integration tests.
 *
 * @details
 * The fluid contains:
 * - species A with mass `1`,
 * - species B with mass `2`,
 * - equal mole fractions,
 * - buffer size `128`.
 *
 * @return Host-owned shared pointer to the constructed fluid.
 *
 * @tparam T Floating-point scalar type.
 */
template <typename T>
atlas::FluidHostPtr<T>
make_test_fluid() {
    const auto species_a = atlas::system::MatrialProperties<T>::builder()
                               .with_mass(T(1))
                               .make_host_shared();
    const auto species_b = atlas::system::MatrialProperties<T>::builder()
                               .with_mass(T(2))
                               .make_host_shared();

    return atlas::system::Fluid<T>::builder()
        .add_species(species_a, T(0.5))
        .add_species(species_b, T(0.5))
        .with_buffer_size(128)
        .make_host_shared();
}

/**
 * @brief Create a simple source configured with one test unit and one volume spawn mode.
 *
 * @return Host-owned shared pointer to the configured source.
 *
 * @tparam T Floating-point scalar type.
 */
template <typename T>
atlas::SourceHostPtr<T>
make_test_source() {
    return atlas::Source<T>::builder()
        .with_units({ *test::make_host_shared_unit<T>() })
        .with_fluid(make_test_fluid<T>())
        .with_spawn_types({ atlas::system::SpawnType::Volume })
        .with_spawn_operator(atlas::system::SpawnOperator<T>(atlas::system::SpawnType::Volume))
        .with_spacing(T(1))
        .make_host_shared();
}

/**
 * @brief Create a simple sink configured with one test unit and one volume despawn mode.
 *
 * @return Host-owned shared pointer to the configured sink.
 *
 * @tparam T Floating-point scalar type.
 */
template <typename T>
atlas::SinkHostPtr<T>
make_test_sink() {
    return atlas::Sink<T>::builder()
        .with_units({ *test::make_host_shared_unit<T>() })
        .with_despawn_types({ atlas::system::DespawnType::Volume })
        .with_despawn_operator(atlas::system::DespawnOperator<T>(atlas::system::DespawnType::Volume))
        .make_host_shared();
}

} // namespace atlas::test

#ifdef ATLAS_ENABLE_VIZKIT

#include <vizkit/layer/geometry/geometry_layer.h>
#include <vizkit/layer/layer.h>

namespace atlas::test {

/**
 * @brief Create a Vizkit-compatible box unit from explicit bounds.
 *
 * @param lower Lower local-space corner of the box.
 * @param upper Upper local-space corner of the box.
 * @return Host-owned shared pointer to the constructed unit.
 *
 * @tparam T Floating-point scalar type.
 */
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

/**
 * @brief Create a Vizkit-compatible translated unit containing a canonical box.
 *
 * @param translation World-space translation applied through the sync object.
 * @return Host-owned shared pointer to the constructed unit.
 *
 * @tparam T Floating-point scalar type.
 */
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

/**
 * @brief Dummy Vizkit layer used for initialization and update callback tests.
 *
 * @details
 * This test double records:
 * - whether @ref init was called,
 * - whether @ref update was called,
 * - the last GLFW window pointer passed in,
 * - the last timestep received.
 *
 * @tparam T Floating-point scalar type.
 */
template <typename T>
class DummyVizkitLayer final : public atlas::vizkit::Layer<T> {
public:
    /**
     * @brief Whether @ref init has been called.
     */
    bool init_called = false;

    /**
     * @brief Whether @ref update has been called.
     */
    bool update_called = false;

    /**
     * @brief Last window pointer received by the layer.
     */
    GLFWwindow* last_window = nullptr;

    /**
     * @brief Last timestep received by the layer.
     */
    T last_dt = T(0);

public:
    /**
     * @brief Record that initialization was requested.
     *
     * @param window GLFW window pointer.
     * @param camera Vizkit camera.
     */
    void
    init(GLFWwindow* window, atlas::vizkit::Camera& camera) override {
        init_called = true;
        last_window = window;
        (void)camera;
    }

    /**
     * @brief Record that an update was requested.
     *
     * @param window GLFW window pointer.
     * @param camera Vizkit camera.
     * @param dt Frame timestep.
     */
    void
    update(GLFWwindow* window, atlas::vizkit::Camera& camera, T dt) override {
        update_called = true;
        last_window   = window;
        last_dt       = dt;
        (void)camera;
    }
};

/**
 * @brief Test geometry layer exposing protected geometry/synchronization state for Vizkit tests.
 *
 * @details
 * This helper derives from @ref atlas::vizkit::GeometryLayer and:
 * - stores a seed position list used by @ref build_geometry,
 * - exposes mutable access to the internal local-position buffer,
 * - exposes the protected `synchronize` method through a public wrapper.
 *
 * @tparam T Floating-point scalar type.
 */
template <typename T>
class TestVizkitGeometryLayer final : public atlas::vizkit::GeometryLayer<T> {
public:
    /**
     * @brief Seed positions returned by @ref build_geometry.
     */
    std::vector<atlas::Vector3<T>> seeded_positions;

public:
    /**
     * @brief Construct the test geometry layer.
     *
     * @param unit Optional unit associated with the geometry layer.
     */
    explicit TestVizkitGeometryLayer(const atlas::UnitHostPtr<T>& unit = nullptr)
        : atlas::vizkit::GeometryLayer<T>(GL_LINES, unit) { }

    /**
     * @brief Return mutable access to the internal local-position buffer.
     *
     * @return Reference to the internal local positions.
     */
    std::vector<atlas::Vector3<T>>&
    local_positions_public() {
        return this->_local_positions;
    }

    /**
     * @brief Public wrapper around the protected synchronize helper.
     *
     * @param world_positions Output world-space positions.
     * @param dt Timestep.
     * @return Result of the underlying synchronization routine.
     */
    bool
    synchronize_public(std::vector<atlas::Vector3<T>>& world_positions, T dt) {
        return this->synchronize(world_positions, dt);
    }

protected:
    /**
     * @brief Populate the geometry layer with the current seeded positions.
     *
     * @param positions Output geometry positions.
     */
    void
    build_geometry(std::vector<atlas::Vector3<T>>& positions) override {
        positions = seeded_positions;
    }
};

} // namespace atlas::test

#endif
