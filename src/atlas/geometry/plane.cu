#include <atlas/geometry/plane.h>

#include <stdexcept>

namespace atlas {

Plane::Builder
Plane::builder() noexcept {
    return Builder {};
}

Plane
Plane::Builder::build() const {
    validate();

    return Plane(_normal, _offset);
}

atlas::host_shared_ptr<Plane>
Plane::Builder::make_host_shared() const {
    return atlas::make_host_shared<Plane>(build());
}

Plane::Builder&
Plane::Builder::with_normal(const Vector3& normal_) noexcept {
    _normal = normal_;
    return *this;
}

Plane::Builder&
Plane::Builder::with_offset(const float offset_) noexcept {
    _offset = offset_;
    return *this;
}

Plane::Builder&
Plane::Builder::with_normal_offset(const Vector3& normal_, const float offset_) noexcept {
    _normal = normal_;
    _offset = offset_;
    return *this;
}

Plane::Builder&
Plane::Builder::with_point_normal(const Vector3& point, const Vector3& normal_) noexcept {
    _normal = normal_;
    _offset = -(normal_.dot(point));
    return *this;
}

void
Plane::Builder::validate() const {
    if (!Plane(_normal, _offset).is_valid()) {
        throw std::runtime_error("Plane::Builder: invalid parameters.");
    }
}

}
