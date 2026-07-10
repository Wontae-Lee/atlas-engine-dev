#include <atlas/geometry/square.h>

#include <stdexcept>
#include <utility>

namespace atlas {

Square::Builder
Square::builder() noexcept {
    return Builder {};
}

Square
Square::Builder::build() const {
    // Reject non-finite, zero-normal, or non-positive-side parameters.
    validate();

    return Square(_center, _normal, _side_length);
}

atlas::host_shared_ptr<Square>
Square::Builder::make_host_shared() const {
    auto square = build();
    return atlas::make_host_shared<Square>(std::move(square));
}

Square::Builder&
Square::Builder::with_center(const Float3& center_) noexcept {
    _center = center_;
    return *this;
}

Square::Builder&
Square::Builder::with_normal(const Float3& normal_) noexcept {
    _normal = normal_;
    return *this;
}

Square::Builder&
Square::Builder::with_side_length(const float side_length_) noexcept {
    _side_length = side_length_;
    return *this;
}

void
Square::Builder::validate() const {
    // Fast path: a fully valid square needs no diagnosis.
    if (Square(_center, _normal, _side_length).is_valid()) {
        return;
    }

    // Otherwise report the first violated invariant with a specific message.
    if (!atlas::isfinite(_center)
        || !atlas::isfinite(_normal)
        || !atlas::isfinite(_side_length)) {
        throw std::runtime_error("Square::Builder: parameters must be finite.");
    }

    if (!(_normal.length_squared() > 0.0f)) {
        throw std::runtime_error("Square::Builder: normal must be non-zero.");
    }

    // Only remaining failure once finite and non-zero-normal are ruled out.
    throw std::runtime_error("Square::Builder: side_length must be positive.");
}

}
