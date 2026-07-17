#include <atlas/geometry/polygonal_prism.h>

#include <stdexcept>

namespace atlas {

PolygonalPrism::Builder PolygonalPrism::builder() noexcept { return Builder {}; }

PolygonalPrism PolygonalPrism::Builder::build() const {
    validate();
    return PolygonalPrism(_center, _side_count, _radius, _height);
}

atlas::host_shared_ptr<PolygonalPrism> PolygonalPrism::Builder::make_host_shared() const {
    return atlas::make_host_shared<PolygonalPrism>(build());
}

PolygonalPrism::Builder& PolygonalPrism::Builder::with_center(const Float3& center_) noexcept {
    _center = center_;
    return *this;
}

PolygonalPrism::Builder& PolygonalPrism::Builder::with_side_count(int side_count_) noexcept {
    _side_count = side_count_;
    return *this;
}

PolygonalPrism::Builder& PolygonalPrism::Builder::with_radius(float radius_) noexcept {
    _radius = radius_;
    return *this;
}

PolygonalPrism::Builder& PolygonalPrism::Builder::with_height(float height_) noexcept {
    _height = height_;
    return *this;
}

void PolygonalPrism::Builder::validate() const {
    if (!PolygonalPrism(_center, _side_count, _radius, _height).is_valid()) {
        throw std::runtime_error("PolygonalPrism::Builder: invalid parameters.");
    }
}

}
