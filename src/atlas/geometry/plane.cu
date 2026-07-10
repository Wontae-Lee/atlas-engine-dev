#include <atlas/geometry/plane.h>

#include <stdexcept>

namespace atlas {

Plane::Builder
Plane::builder() noexcept {
    return Builder {};
}

Plane
Plane::Builder::build() const {
    // Reject non-finite or zero-length normals before constructing.
    validate();

    return Plane(_normal, _offset);
}

atlas::host_shared_ptr<Plane>
Plane::Builder::make_host_shared() const {
    return atlas::make_host_shared<Plane>(build());
}

Plane::Builder&
Plane::Builder::with_normal(const Float3& normal_) noexcept {
    _normal = normal_;
    return *this;
}

Plane::Builder&
Plane::Builder::with_offset(const float offset_) noexcept {
    _offset = offset_;
    return *this;
}

Plane::Builder&
Plane::Builder::with_normal_offset(const Float3& normal_, const float offset_) noexcept {
    _normal = normal_;
    _offset = offset_;
    return *this;
}

Plane::Builder&
Plane::Builder::with_point_normal(const Float3& point, const Float3& normal_) noexcept {
    _normal = normal_;
    // Encode "plane passes through point" as the Hessian offset.
    _offset = -(normal_.dot(point));
    return *this;
}

void
Plane::Builder::validate() const {
    // Reuse the Plane's own validity invariant on a throwaway instance.
    if (!Plane(_normal, _offset).is_valid()) {
        throw std::runtime_error("Plane::Builder: invalid parameters.");
    }
}

}
