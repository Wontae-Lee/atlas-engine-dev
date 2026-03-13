#pragma once

#ifdef ATLAS_ENABLE_VIZKIT

namespace atlas::vizkit {

template <typename T>
BoxLayer<T>::BoxLayer(const atlas::UnitHostPtr<T>& unit)
    // GL_LINES means OpenGL interprets every adjacent vertex pair
    // (v0, v1), (v2, v3), ... as one independent line segment.
    // A box wireframe therefore needs 12 edges * 2 endpoints = 24 vertices.
    : GeometryLayer<T>(GL_LINES, unit) { }

template <typename T>
typename BoxLayer<T>::Builder
BoxLayer<T>::builder() noexcept {
    return Builder {};
}

template <typename T>
void
BoxLayer<T>::append_local_box_lines(
    const Vector3<T>& min_corner,
    const Vector3<T>& max_corner,
    std::vector<Vector3<T>>& positions) const {

    // Rebuild the wireframe from scratch.
    // This function produces an explicit line list rather than indexed edges,
    // because the base rendering path uses glDrawArrays instead of an index
    // buffer. Duplicating vertices keeps the OpenGL submission path simple.
    positions.clear();
    positions.reserve(24);

    // Enumerate the 8 corners of the axis-aligned box.
    //
    // Naming convention:
    // - first bit  : x chooses min/max
    // - second bit : y chooses min/max
    // - third bit  : z chooses min/max
    //
    // Example:
    //   v101 = (x=max, y=min, z=max)
    //
    // Because the box is axis-aligned in local space, corners are obtained by
    // simple component selection rather than any rotation or basis transform.
    Vector3<T> v000(min_corner.x, min_corner.y, min_corner.z);
    Vector3<T> v100(max_corner.x, min_corner.y, min_corner.z);
    Vector3<T> v110(max_corner.x, max_corner.y, min_corner.z);
    Vector3<T> v010(min_corner.x, max_corner.y, min_corner.z);

    Vector3<T> v001(min_corner.x, min_corner.y, max_corner.z);
    Vector3<T> v101(max_corner.x, min_corner.y, max_corner.z);
    Vector3<T> v111(max_corner.x, max_corner.y, max_corner.z);
    Vector3<T> v011(min_corner.x, max_corner.y, max_corner.z);

    // Bottom face edges (z = min).
    // Each pair defines one segment in the line list.
    positions.push_back(v000);
    positions.push_back(v100);
    positions.push_back(v100);
    positions.push_back(v110);
    positions.push_back(v110);
    positions.push_back(v010);
    positions.push_back(v010);
    positions.push_back(v000);

    // Top face edges (z = max).
    positions.push_back(v001);
    positions.push_back(v101);
    positions.push_back(v101);
    positions.push_back(v111);
    positions.push_back(v111);
    positions.push_back(v011);
    positions.push_back(v011);
    positions.push_back(v001);

    // Vertical edges connecting bottom and top faces.
    //
    // Geometrically, these are the segments parallel to the local Z axis that
    // close the rectangular prism.
    positions.push_back(v000);
    positions.push_back(v001);
    positions.push_back(v100);
    positions.push_back(v101);
    positions.push_back(v110);
    positions.push_back(v111);
    positions.push_back(v010);
    positions.push_back(v011);
}

template <typename T>
void
BoxLayer<T>::build_geometry(std::vector<Vector3<T>>& positions) {

    // The derived layer is responsible only for producing local-space geometry.
    // World-space placement happens later in GeometryLayer::synchronize via the
    // unit's SyncOperator.
    positions.clear();
    if (!this->_unit) return;

    const auto& query = this->_unit->query_operator();

    // A unit may hold many possible query variants. This layer renders only the
    // box variant, so mismatched types produce no geometry.
    if (query.type != atlas::geometry::GeometryType::Box) return;

    const auto& box = query.box;

    // lower_corner and upper_corner define the two opposite corners of an
    // axis-aligned bounding box in local coordinates.
    if (!box.lower_corner || !box.upper_corner) return;

    append_local_box_lines(*box.lower_corner, *box.upper_corner, positions);
}

template <typename T>
typename BoxLayer<T>::Builder&
BoxLayer<T>::Builder::with_unit(const atlas::UnitHostPtr<T>& unit) noexcept {
    _unit = unit;
    return *this;
}

template <typename T>
void
BoxLayer<T>::Builder::validate() const {
    if (!_unit) throw std::runtime_error("BoxLayer: unit null");
}

template <typename T>
BoxLayer<T>
BoxLayer<T>::Builder::build() const {
    validate();
    return BoxLayer(_unit);
}

template <typename T>
std::shared_ptr<BoxLayer<T>>
BoxLayer<T>::Builder::make_shared() const {
    validate();
    return std::make_shared<BoxLayer>(_unit);
}

}

#endif
