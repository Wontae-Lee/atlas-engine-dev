#include <atlas/geometry/sphere.h>

#include <stdexcept>

namespace atlas {

Sphere::Builder
Sphere::builder() noexcept {
    return Builder {};
}

Sphere
Sphere::Builder::build() const {
    validate();

    return Sphere(_center, _radius);
}

atlas::host_shared_ptr<Sphere>
Sphere::Builder::make_host_shared() const {
    return atlas::make_host_shared<Sphere>(build());
}

Sphere::Builder&
Sphere::Builder::with_center(const Vector3& c) noexcept {
    _center = c;
    return *this;
}

Sphere::Builder&
Sphere::Builder::with_radius(const float r) noexcept {
    _radius = r;
    return *this;
}

void
Sphere::Builder::validate() const {
    if (!Sphere(_center, _radius).is_valid()) {
        throw std::runtime_error("Sphere::Builder: invalid parameters.");
    }
}

}
