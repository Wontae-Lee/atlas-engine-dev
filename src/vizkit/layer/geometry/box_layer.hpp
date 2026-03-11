#pragma once

#ifdef ATLAS_ENABLE_VIZKIT

namespace atlas::vizkit {

template <typename T>
BoxLayer<T>::BoxLayer(const atlas::UnitHostPtr<T>& unit)
    : GeometryLayer<T>(GL_LINES, unit) {}

template <typename T>
typename BoxLayer<T>::Builder
BoxLayer<T>::builder() noexcept {
    return Builder{};
}

template <typename T>
void
BoxLayer<T>::append_local_box_lines(
    const Vector3<T>& min_corner,
    const Vector3<T>& max_corner,
    std::vector<Vector3<T>>& positions) const {

    positions.clear();
    positions.reserve(24);

    Vector3<T> v000(min_corner.x,min_corner.y,min_corner.z);
    Vector3<T> v100(max_corner.x,min_corner.y,min_corner.z);
    Vector3<T> v110(max_corner.x,max_corner.y,min_corner.z);
    Vector3<T> v010(min_corner.x,max_corner.y,min_corner.z);

    Vector3<T> v001(min_corner.x,min_corner.y,max_corner.z);
    Vector3<T> v101(max_corner.x,min_corner.y,max_corner.z);
    Vector3<T> v111(max_corner.x,max_corner.y,max_corner.z);
    Vector3<T> v011(min_corner.x,max_corner.y,max_corner.z);

    positions.push_back(v000); positions.push_back(v100);
    positions.push_back(v100); positions.push_back(v110);
    positions.push_back(v110); positions.push_back(v010);
    positions.push_back(v010); positions.push_back(v000);

    positions.push_back(v001); positions.push_back(v101);
    positions.push_back(v101); positions.push_back(v111);
    positions.push_back(v111); positions.push_back(v011);
    positions.push_back(v011); positions.push_back(v001);

    positions.push_back(v000); positions.push_back(v001);
    positions.push_back(v100); positions.push_back(v101);
    positions.push_back(v110); positions.push_back(v111);
    positions.push_back(v010); positions.push_back(v011);
}

template <typename T>
void
BoxLayer<T>::build_geometry(std::vector<Vector3<T>>& positions) {

    positions.clear();
    if (!this->_unit) return;

    const auto& query = this->_unit->query_operator();
    if (query.type != atlas::geometry::GeometryType::Box) return;

    const auto& box = query.box;
    if (!box.lower_corner || !box.upper_corner) return;

    append_local_box_lines(*box.lower_corner,*box.upper_corner,positions);
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
    if(!_unit) throw std::runtime_error("BoxLayer: unit null");
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