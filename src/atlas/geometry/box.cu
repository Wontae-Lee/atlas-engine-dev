#include <atlas/geometry/box.h>

#include <stdexcept>

namespace atlas {

Box::Builder
Box::builder() noexcept {
    return Builder {};
}

Box
Box::Builder::build() const {
    validate();

    return Box(_lower_corner, _upper_corner);
}

atlas::host_shared_ptr<Box>
Box::Builder::make_host_shared() const {
    return atlas::make_host_shared<Box>(build());
}

Box::Builder&
Box::Builder::with_lower_corner(const Vector3& lower_corner_) noexcept {
    _lower_corner = lower_corner_;
    return *this;
}

Box::Builder&
Box::Builder::with_upper_corner(const Vector3& upper_corner_) noexcept {
    _upper_corner = upper_corner_;
    return *this;
}

void
Box::Builder::validate() const {
    if (!Box(_lower_corner, _upper_corner).is_valid()) {
        throw std::runtime_error("Box::Builder: invalid parameters.");
    }
}

}
