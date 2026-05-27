#pragma once

#include <atlas/core/macros.h>
#include <atlas/material/material_properties.h>
#include <atlas/math/vector/vector3.h>
#include <atlas/solver/dsmc/variable_hard_sphere_kernel.h>

#include <cstddef>

namespace atlas::system {

template <typename T>
class PiclasVhsKernel final {
public:
    ATLAS_ALL_DEVICE ATLAS_NODISCARD ATLAS_FORCE_INLINE static T
    cross_section_factor(const MaterialProperties<T>& lhs,
                         const MaterialProperties<T>& rhs) noexcept;

    ATLAS_ALL_DEVICE ATLAS_NODISCARD ATLAS_FORCE_INLINE static T
    sigma_g(const MaterialProperties<T>& lhs,
            const MaterialProperties<T>& rhs,
            T relative_speed_squared) noexcept;

    ATLAS_ALL_DEVICE ATLAS_NODISCARD ATLAS_FORCE_INLINE T
    sigma_g(const MaterialProperties<T>* properties_ptr,
            std::size_t species_i,
            std::size_t species_j,
            T relative_speed_squared) const noexcept;

    ATLAS_ALL_DEVICE ATLAS_NODISCARD ATLAS_FORCE_INLINE static T
    collision_probability(T sigma_g,
                          T species_count_i,
                          T species_count_j,
                          bool same_species,
                          int pair_case_count,
                          T statistical_weight,
                          T dt,
                          T volume) noexcept;

    ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE void
    operator()(Vector3<T>& lhs_velocity,
               Vector3<T>& rhs_velocity,
               const MaterialProperties<T>& lhs,
               const MaterialProperties<T>& rhs) const noexcept;
};

} // namespace atlas::system

namespace atlas {

template <typename T>
using PiclasVhsKernel = atlas::system::PiclasVhsKernel<T>;

} // namespace atlas

#include <atlas/solver/dsmc/piclas/piclas_vhs_kernel.hpp>
