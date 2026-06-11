#pragma once

namespace atlas::spatial {

template <typename T, typename TransformPoint>
ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE
AxisAlignedBoundingBox<T>
transform_aabb(const AxisAlignedBoundingBox<T>& bound,
               TransformPoint transform) noexcept {
    AxisAlignedBoundingBox<T> transformed {};
    if (!bound.is_valid()) {
        return transformed;
    }

    for (int corner = 0; corner < 8; ++corner) {
        transformed.merge(transform(bound.corner(static_cast<std::size_t>(corner))));
    }

    return transformed;
}

} // namespace atlas::spatial
