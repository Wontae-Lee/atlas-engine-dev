#include <atlas/sync/sync.h>

#include <stdexcept>
#include <utility>

namespace atlas {

Sync::Builder
Sync::builder() noexcept {
    return Builder {};
}

Sync::Builder&
Sync::Builder::with_rigid_pose(const Vector3& translation_,
                               const Quaternion& orientation_) noexcept {
    _translation = translation_;
    _orientation = orientation_;
    return *this;
}

void
Sync::Builder::validate() const {
    if (!atlas::isfinite(_translation)) {
        throw std::runtime_error(
            "Sync::Builder: translation contains non-finite values.");
    }

    if (!atlas::isfinite(_orientation)) {
        throw std::runtime_error(
            "Sync::Builder: orientation contains non-finite values.");
    }

    if (_orientation.w == 0.0f
        && _orientation.x == 0.0f
        && _orientation.y == 0.0f
        && _orientation.z == 0.0f) {
        throw std::runtime_error(
            "Sync::Builder: zero quaternion is invalid.");
    }
}

Sync
Sync::Builder::build() const {
    validate();

    return Sync(_translation, _orientation);
}

atlas::host_shared_ptr<Sync>
Sync::Builder::make_host_shared() const {
    return atlas::make_host_shared<Sync>(build());
}

}
