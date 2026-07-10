#include <atlas/geometry/cylinder.h>

#include <stdexcept>

namespace atlas {

Cylinder::Builder
Cylinder::builder() noexcept {
    return Builder {};
}

Cylinder
Cylinder::Builder::build() const {
    // Positive radius and height are enforced here.
    validate();

    Cylinder c(_center, _radius, _height);
    // The constructor always makes a capped cylinder, so apply the flag after.
    c.open = _open;

    return c;
}

atlas::host_shared_ptr<Cylinder>
Cylinder::Builder::make_host_shared() const {
    return atlas::make_host_shared<Cylinder>(build());
}

Cylinder::Builder&
Cylinder::Builder::with_center(const Float3& center_) noexcept {
    _center = center_;
    return *this;
}

Cylinder::Builder&
Cylinder::Builder::with_radius(const float radius_) noexcept {
    _radius = radius_;
    return *this;
}

Cylinder::Builder&
Cylinder::Builder::with_height(const float height_) noexcept {
    _height = height_;
    return *this;
}

Cylinder::Builder&
Cylinder::Builder::with_open(const bool open_) noexcept {
    _open = open_;
    return *this;
}

void
Cylinder::Builder::validate() const {
    // The open flag never affects validity, so a capped probe instance suffices.
    if (!Cylinder(_center, _radius, _height).is_valid()) {
        throw std::runtime_error("Cylinder::Builder: invalid parameters.");
    }
}

}
