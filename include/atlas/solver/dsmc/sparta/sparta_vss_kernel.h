#pragma once

#include <atlas/core/macros.h>
#include <atlas/material/material_properties.h>
#include <atlas/math/vector/vector3.h>
#include <atlas/solver/dsmc/variable_soft_sphere_kernel.h>

#include <cstddef>

namespace atlas::system {

template <typename T>
class SpartaVssKernel final {
public:
    ATLAS_ALL_DEVICE ATLAS_NODISCARD ATLAS_FORCE_INLINE static T
    collision_frequency_factor(const MaterialProperties<T>& lhs,
                               const MaterialProperties<T>& rhs) noexcept;

    ATLAS_ALL_DEVICE ATLAS_NODISCARD ATLAS_FORCE_INLINE static T
    collision_frequency(const MaterialProperties<T>& lhs,
                        const MaterialProperties<T>& rhs,
                        T relative_speed_squared) noexcept;

    ATLAS_ALL_DEVICE ATLAS_NODISCARD ATLAS_FORCE_INLINE T
    collision_frequency(const MaterialProperties<T>* properties_ptr,
                        std::size_t species_i,
                        std::size_t species_j,
                        T relative_speed_squared) const noexcept;

    ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE void
    operator()(Vector3<T>& lhs_velocity,
               Vector3<T>& rhs_velocity,
               const MaterialProperties<T>& lhs,
               const MaterialProperties<T>& rhs) const noexcept;
};

} // namespace atlas::system

namespace atlas {

template <typename T>
using SpartaVssKernel = atlas::system::SpartaVssKernel<T>;

} // namespace atlas

#include <atlas/solver/dsmc/sparta/sparta_vss_kernel.hpp>
