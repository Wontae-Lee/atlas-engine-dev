#pragma once
#include <limits>
namespace atlas {

namespace detail {

    template <typename T>
    using GeometryOperatorVariant = DeviceVariant<
        GeometryOperator<T>,
        GeometryType,
        GeometryType::Sphere,
        DeviceVariantCase<
            GeometryOperator<T>,
            GeometryType,
            GeometryType::Box,
            BoxGeometryOperator<T>,
            &GeometryOperator<T>::box>,
        DeviceVariantCase<
            GeometryOperator<T>,
            GeometryType,
            GeometryType::Circle,
            CircleGeometryOperator<T>,
            &GeometryOperator<T>::circle>,
        DeviceVariantCase<
            GeometryOperator<T>,
            GeometryType,
            GeometryType::Cylinder,
            CylinderGeometryOperator<T>,
            &GeometryOperator<T>::cylinder>,
        DeviceVariantCase<
            GeometryOperator<T>,
            GeometryType,
            GeometryType::Plane,
            PlaneGeometryOperator<T>,
            &GeometryOperator<T>::plane>,
        DeviceVariantCase<
            GeometryOperator<T>,
            GeometryType,
            GeometryType::Sphere,
            SphereGeometryOperator<T>,
            &GeometryOperator<T>::sphere>,
        DeviceVariantCase<
            GeometryOperator<T>,
            GeometryType,
            GeometryType::Square,
            SquareGeometryOperator<T>,
            &GeometryOperator<T>::square>,
        DeviceVariantCase<
            GeometryOperator<T>,
            GeometryType,
            GeometryType::Triangle,
            TriangleGeometryOperator<T>,
            &GeometryOperator<T>::triangle>,
        DeviceVariantCase<
            GeometryOperator<T>,
            GeometryType,
            GeometryType::TriangleMesh,
            TriangleMeshGeometryOperator<T>,
            &GeometryOperator<T>::triangle_mesh>>;

}

template <typename T>
GeometryOperator<T>::GeometryOperator() noexcept {
    detail::GeometryOperatorVariant<T>::construct(*this, GeometryType::Sphere);
}

template <typename T>
GeometryOperator<T>::GeometryOperator(const GeometryOperator& other) noexcept {
    detail::GeometryOperatorVariant<T>::copy_construct(*this, other);
}

template <typename T>
GeometryOperator<T>&
GeometryOperator<T>::operator=(const GeometryOperator& other) noexcept {
    detail::GeometryOperatorVariant<T>::assign(*this, other);
    return *this;
}

template <typename T>
GeometryOperator<T>::~GeometryOperator() noexcept {
    detail::GeometryOperatorVariant<T>::destroy(*this);
}

template <typename T>
GeometryOperator<T>::GeometryOperator(const BoxGeometryOperator<T>& op) {
    detail::GeometryOperatorVariant<T>::construct_payload(*this, op);
}

template <typename T>
GeometryOperator<T>::GeometryOperator(const CircleGeometryOperator<T>& op) {
    detail::GeometryOperatorVariant<T>::construct_payload(*this, op);
}

template <typename T>
GeometryOperator<T>::GeometryOperator(const CylinderGeometryOperator<T>& op) {
    detail::GeometryOperatorVariant<T>::construct_payload(*this, op);
}

template <typename T>
GeometryOperator<T>::GeometryOperator(const PlaneGeometryOperator<T>& op) {
    detail::GeometryOperatorVariant<T>::construct_payload(*this, op);
}

template <typename T>
GeometryOperator<T>::GeometryOperator(const SphereGeometryOperator<T>& op) {
    detail::GeometryOperatorVariant<T>::construct_payload(*this, op);
}

template <typename T>
GeometryOperator<T>::GeometryOperator(const SquareGeometryOperator<T>& op) {
    detail::GeometryOperatorVariant<T>::construct_payload(*this, op);
}

template <typename T>
GeometryOperator<T>::GeometryOperator(const TriangleGeometryOperator<T>& op) {
    detail::GeometryOperatorVariant<T>::construct_payload(*this, op);
}

template <typename T>
GeometryOperator<T>::GeometryOperator(const TriangleMeshGeometryOperator<T>& op) {
    detail::GeometryOperatorVariant<T>::construct_payload(*this, op);
}

template <typename T>
atlas::Vector<T, 3>
GeometryOperator<T>::closest_point(const atlas::Vector<T, 3>& p) const noexcept {
    return detail::GeometryOperatorVariant<T>::visit(
        *this,
        [&] ATLAS_ALL_DEVICE (const auto& geometry) noexcept { return geometry.closest_point(p); },
        p);
}

template <typename T>
atlas::Vector<T, 3>
GeometryOperator<T>::closest_normal(const atlas::Vector<T, 3>& p) const noexcept {
    return detail::GeometryOperatorVariant<T>::visit(
        *this,
        [&] ATLAS_ALL_DEVICE (const auto& geometry) noexcept { return geometry.closest_normal(p); },
        atlas::Vector<T, 3>(T(0), T(0), T(0)));
}

template <typename T>
T
GeometryOperator<T>::signed_distance(const atlas::Vector<T, 3>& p) const noexcept {
    return detail::GeometryOperatorVariant<T>::visit(
        *this,
        [&] ATLAS_ALL_DEVICE (const auto& geometry) noexcept { return geometry.signed_distance(p); },
        std::numeric_limits<T>::infinity());
}

template <typename T>
bool
GeometryOperator<T>::is_inside(const atlas::Vector<T, 3>& p, const T tolerance) const noexcept {
    return detail::GeometryOperatorVariant<T>::visit(
        *this,
        [&] ATLAS_ALL_DEVICE (const auto& geometry) noexcept { return geometry.is_inside(p, tolerance); },
        false);
}

template <typename T>
bool
GeometryOperator<T>::is_on_surface(const atlas::Vector<T, 3>& p, const T tolerance) const noexcept {
    return detail::GeometryOperatorVariant<T>::visit(
        *this,
        [&] ATLAS_ALL_DEVICE (const auto& geometry) noexcept { return geometry.is_on_surface(p, tolerance); },
        false);
}

template <typename T>
atlas::Vector<T, 3>
GeometryOperator<T>::centroid() const noexcept {
    return detail::GeometryOperatorVariant<T>::visit(
        *this,
        [&] ATLAS_ALL_DEVICE (const auto& geometry) noexcept { return geometry.centroid(); },
        atlas::Vector<T, 3>(T(0), T(0), T(0)));
}

template <typename T>
atlas::AxisAlignedBoundingBox<T>
GeometryOperator<T>::bound() const noexcept {
    return detail::GeometryOperatorVariant<T>::visit(
        *this,
        [&] ATLAS_ALL_DEVICE (const auto& geometry) noexcept { return geometry.bound(); },
        atlas::AxisAlignedBoundingBox<T>());
}

template <typename T>
bool
GeometryOperator<T>::is_valid() const noexcept {
    return detail::GeometryOperatorVariant<T>::visit(
        *this,
        [&] ATLAS_ALL_DEVICE (const auto& geometry) noexcept { return geometry.is_valid(); },
        false);
}

template <typename T>
HitSurface<T>
GeometryOperator<T>::trace(const atlas::Ray<T>& ray) const noexcept {
    return detail::GeometryOperatorVariant<T>::visit(
        *this,
        [&] ATLAS_ALL_DEVICE (const auto& geometry) noexcept { return geometry.trace(ray); },
        HitSurface<T> {});
}

template <typename T>
HitSurface<T>
GeometryOperator<T>::operator()(const atlas::Ray<T>& ray) const noexcept {
    return trace(ray);
}

}