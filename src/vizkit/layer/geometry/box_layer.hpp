#pragma once

#ifdef ATLAS_ENABLE_VIZKIT

namespace atlas::vizkit {

template <typename T>
BoxLayer<T>::BoxLayer(const atlas::UnitHostPtr<T>& unit)
    : GeometryLayer<T>(GL_LINES, unit) {
    // Construct a box visualization layer from a unit.
    //
    // Rendering mode:
    // - GL_LINES
    //
    // This means the box is rendered as a wireframe line set rather than
    // filled triangles.
    //
    // The provided unit is forwarded to the GeometryLayer base class and is
    // later queried for:
    // - geometry type
    // - box bounds
    // - transform information inherited from the unit abstraction
}

template <typename T>
typename BoxLayer<T>::Builder
BoxLayer<T>::builder() noexcept {
    // Return a fresh builder for staged BoxLayer construction.
    return Builder {};
}

template <typename T>
void
BoxLayer<T>::append_local_box_lines(
    const Vector3<T>& min_corner,
    const Vector3<T>& max_corner,
    std::vector<Vector3<T>>& positions) const {

    // Rebuild the line-vertex array from scratch.
    positions.clear();

    // A wireframe box has 12 edges.
    // Since each edge is represented by two endpoints in GL_LINES mode,
    // the total number of vertices is:
    //   12 * 2 = 24
    positions.reserve(24);

    // Define the 8 local-space box corners.
    //
    // Naming convention:
    // - first bit  : x side (0=min, 1=max)
    // - second bit : y side (0=min, 1=max)
    // - third bit  : z side (0=min, 1=max)
    //
    // Example:
    // - v000 = (xmin, ymin, zmin)
    // - v111 = (xmax, ymax, zmax)
    Vector3<T> v000(min_corner.x, min_corner.y, min_corner.z);
    Vector3<T> v100(max_corner.x, min_corner.y, min_corner.z);
    Vector3<T> v110(max_corner.x, max_corner.y, min_corner.z);
    Vector3<T> v010(min_corner.x, max_corner.y, min_corner.z);

    Vector3<T> v001(min_corner.x, min_corner.y, max_corner.z);
    Vector3<T> v101(max_corner.x, min_corner.y, max_corner.z);
    Vector3<T> v111(max_corner.x, max_corner.y, max_corner.z);
    Vector3<T> v011(min_corner.x, max_corner.y, max_corner.z);

    // Append the 4 bottom-face edges (z = min).
    //
    // Edge order:
    // v000 -> v100
    // v100 -> v110
    // v110 -> v010
    // v010 -> v000
    positions.push_back(v000);
    positions.push_back(v100);

    positions.push_back(v100);
    positions.push_back(v110);

    positions.push_back(v110);
    positions.push_back(v010);

    positions.push_back(v010);
    positions.push_back(v000);

    // Append the 4 top-face edges (z = max).
    //
    // Edge order:
    // v001 -> v101
    // v101 -> v111
    // v111 -> v011
    // v011 -> v001
    positions.push_back(v001);
    positions.push_back(v101);

    positions.push_back(v101);
    positions.push_back(v111);

    positions.push_back(v111);
    positions.push_back(v011);

    positions.push_back(v011);
    positions.push_back(v001);

    // Append the 4 vertical edges connecting bottom and top faces.
    //
    // Edge order:
    // v000 -> v001
    // v100 -> v101
    // v110 -> v111
    // v010 -> v011
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

    // Always rebuild the output geometry buffer from scratch.
    positions.clear();

    // No unit means there is no source geometry to visualize.
    if (!this->_unit) return;

    // Read the unit's geometry query operator.
    const auto& query = this->_unit->geometry_operator();

    // This layer only knows how to render box geometry.
    // If the unit geometry is not a box, leave the output empty.
    if (query.type != atlas::geometry::GeometryType::Box) return;

    // Access the concrete box operator stored inside the tagged geometry union.
    const auto& box = query.box;

    // The box operator must expose valid lower/upper corner pointers in order
    // to generate line vertices.
    if (!box.lower_corner || !box.upper_corner) return;

    // Build the local-space wireframe line set from the box corners.
    append_local_box_lines(*box.lower_corner, *box.upper_corner, positions);
}

template <typename T>
typename BoxLayer<T>::Builder&
BoxLayer<T>::Builder::with_unit(const atlas::UnitHostPtr<T>& unit) noexcept {
    // Stage the unit that will provide the box geometry for visualization.
    _unit = unit;
    return *this;
}

template <typename T>
void
BoxLayer<T>::Builder::validate() const {
    // A BoxLayer cannot be built without a source unit.
    if (!_unit) throw std::runtime_error("BoxLayer: unit null");
}

template <typename T>
BoxLayer<T>
BoxLayer<T>::Builder::build() const {
    // Validate builder state before constructing the final layer by value.
    validate();
    return BoxLayer(_unit);
}

template <typename T>
std::shared_ptr<BoxLayer<T>>
BoxLayer<T>::Builder::make_shared() const {
    // Validate builder state before constructing the final layer in shared storage.
    validate();
    return std::make_shared<BoxLayer>(_unit);
}

} // namespace atlas::vizkit

#endif