#include <atlas/geometry/circle.h>

#include <stdexcept>

namespace atlas {

Circle::Builder
Circle::builder() noexcept {
    return Builder {};
}

Circle
Circle::Builder::build() const {
    validate();

    return Circle(_center, _normal, _radius);
}

atlas::host_shared_ptr<Circle>
Circle::Builder::make_host_shared() const {
    return atlas::make_host_shared<Circle>(build());
}

Circle::Builder&
Circle::Builder::with_center(const Vector3& center_) noexcept {
    _center = center_;
    return *this;
}

Circle::Builder&
Circle::Builder::with_normal(const Vector3& normal_) noexcept {
    _normal = normal_;
    return *this;
}

Circle::Builder&
Circle::Builder::with_radius(const float radius_) noexcept {
    _radius = radius_;
    return *this;
}

void
Circle::Builder::validate() const {
    if (!Circle(_center, _normal, _radius).is_valid()) {
        throw std::runtime_error("Circle::Builder: invalid parameters.");
    }
}

}
