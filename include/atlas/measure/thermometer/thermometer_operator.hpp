#pragma once

namespace atlas::system {

template <typename T>
void
VarianceThermometerOperator<T>::measure(const DomainDeviceProbe<T>& domain,
                                        const SpatialHashingProbe<T>& searcher,
                                        const FluidDeviceProbe<T>& particle,
                                        const MeasureModeType measure_mode) const {
    // Prepare an optional temporary field buffer.
    //
    // Why this is needed:
    // - MeasureModeType::Field or MeasureModeType::All updates the domain field directly
    // - MeasureModeType::Fluid needs a field-like temperature source only as an intermediate
    // - in Fluid-only mode, writing into the real domain field would be an unwanted side effect
    DeviceBuffer<T> scratch_field {};

    // Pointer to the field-temperature buffer that this operator will use.
    //
    // It points to:
    // - domain.field_temperature when field updates are allowed
    // - scratch_field when only particle temperatures should be updated
    T* field = nullptr;

    // Reuse the real domain field only when the selected mode explicitly allows
    // field-side temperature updates.
    if (measure_mode == MeasureModeType::Field || measure_mode == MeasureModeType::All) {
        field = domain.field_temperature;
    } else {
        // In Fluid-only mode, allocate a temporary per-cell buffer so the cell
        // temperatures can still be computed and sampled by particles without
        // mutating the domain state.
        scratch_field.resize(static_cast<std::size_t>(domain.num_of_cells));
        field = atlas::raw_pointer_cast(scratch_field.data());
    }

    // Cache the particle position array pointer for repeated device access.
    const Vector3<T>* const position = particle.pos;

    // Cache the particle velocity array pointer.
    //
    // Variance-based temperature is derived from velocity fluctuations relative
    // to the local mean velocity.
    const Vector3<T>* const velocity = particle.vel;

    // Cache the particle property table pointer.
    //
    // This is used to read species-dependent particle mass values.
    const MatrialProperties<T>* const particle_property = particle.particle_property;

    // Cache the particle temperature array pointer.
    //
    // This is only required when particle-side temperatures must be written.
    T* const particle_temperature = particle.temperature;

    // Cache the per-particle species-index array.
    //
    // Each particle refers to a species entry in particle_property.
    const size_t* const species = particle.species;

    // Cache the searcher's sorted particle-index array.
    //
    // For each cell range [cell_start[cell], cell_end[cell]), this array stores
    // the indices of the particles belonging to that cell.
    const int* const indices = searcher.indices;

    // Cache the array storing the first sorted-particle slot for each cell.
    const int* const cell_start = searcher.cell_start;

    // Cache the array storing the one-past-the-end sorted-particle slot for each cell.
    const int* const cell_end = searcher.cell_end;

    // Cache grid resolution for repeated coordinate-to-cell mapping.
    const Vector3<int> grid_size = domain.grid_size;

    // Cache domain lower corner used to convert world-space particle positions
    // into grid-relative coordinates.
    const Vector3<T> lower_corner = domain.lower_corner;

    // Cache inverse cell size.
    //
    // Multiplying by inv_h converts world-space distance into cell-space distance.
    const T inv_h = domain.inv_h;

    // Cache total number of cells in the domain.
    const int num_of_cells = domain.num_of_cells;

    // Cache number of active particles.
    const int particle_count = particle.particle_count;

    // Precompute whether particle temperatures should be updated after the
    // per-cell variance temperature has been computed.
    const bool update_fluid = measure_mode == MeasureModeType::Fluid || measure_mode == MeasureModeType::All;

    // Validate all essential runtime pointers and counts before launching any device work.
    //
    // Conditions checked:
    // - field buffer must exist
    // - particle position / velocity / property data must exist
    // - species array and searcher arrays must exist
    // - cell count must be positive
    // - particle count must not be negative
    // - particle temperature output array must exist when fluid update is requested
    if (field == nullptr || position == nullptr || velocity == nullptr || particle_property == nullptr
        || species == nullptr || indices == nullptr || cell_start == nullptr || cell_end == nullptr
        || num_of_cells <= 0 || particle_count < 0
        || (update_fluid && particle_temperature == nullptr)) {
        return;
    }

    // Compute one temperature value per cell from the local velocity variance.
    //
    // High-level workflow per cell:
    // 1. gather all particles belonging to the cell
    // 2. compute the mass-weighted mean velocity
    // 3. accumulate thermal kinetic energy around that mean
    // 4. convert that thermal energy into temperature
    atlas::parallel_for<ExecutionPolicy::device>(
        0,
        num_of_cells,
        [=] ATLAS_DEVICE(const int cell) {
            // Read the first particle slot belonging to this cell.
            const int begin = cell_start[cell];

            // A negative begin index means that the cell is empty.
            //
            // In that case, define the cell temperature as zero.
            if (begin < 0) {
                field[cell] = T(0);
                return;
            }

            // Read the one-past-the-end slot for this cell.
            const int end = cell_end[cell];

            // Accumulate the numerator of the mass-weighted mean velocity:
            //   sum_i (m_i * v_i)
            Vector3<T> mean_velocity { T(0), T(0), T(0) };

            // Accumulate the denominator of the mass-weighted mean velocity:
            //   sum_i m_i
            T momentum_weight_sum = T(0);

            // Count valid particles in the cell.
            int count = 0;

            // First pass over the particles in the cell:
            // - validate particle indices
            // - read each particle mass from its species entry
            // - accumulate mass-weighted velocity
            // - accumulate total mass
            // - count contributors
            for (int k = begin; k < end; ++k) {
                // Recover the original particle index from the searcher.
                const int particle_index = indices[k];

                // Skip any invalid or stale particle indices defensively.
                if (particle_index < 0 || particle_index >= particle_count) continue;

                // Read the species index of the current particle.
                const size_t species_index = species[particle_index];

                // Read the species mass used for momentum weighting.
                const T mass = particle_property[species_index].mass;

                // Accumulate mass-weighted velocity.
                mean_velocity += velocity[particle_index] * mass;

                // Accumulate total mass.
                momentum_weight_sum += mass;

                // Count one more valid particle.
                ++count;
            }

            // If the cell has no valid particles, or total mass is not positive,
            // no variance temperature can be computed.
            if (count <= 0 || !(momentum_weight_sum > T(0))) {
                field[cell] = T(0);
                return;
            }

            // Finish the mass-weighted mean velocity computation:
            //   u = [sum_i m_i v_i] / [sum_i m_i]
            mean_velocity /= momentum_weight_sum;

            // Accumulate thermal kinetic energy relative to the local mean velocity:
            //   sum_i m_i |v_i - u|^2
            T thermal_energy_sum = T(0);

            // Second pass over the particles in the cell:
            // - compute each particle's fluctuating velocity
            // - accumulate its mass-weighted squared fluctuation
            for (int k = begin; k < end; ++k) {
                // Recover the original particle index from the searcher.
                const int particle_index = indices[k];

                // Skip invalid or stale particle indices defensively.
                if (particle_index < 0 || particle_index >= particle_count) continue;

                // Read the particle's species index.
                const size_t species_index = species[particle_index];

                // Read the corresponding species mass.
                const T mass = particle_property[species_index].mass;

                // Compute velocity fluctuation relative to the cell mean.
                const Vector3<T> dv = velocity[particle_index] - mean_velocity;

                // Accumulate mass-weighted squared fluctuation:
                //   m * (dv_x^2 + dv_y^2 + dv_z^2)
                thermal_energy_sum += mass * (dv.x * dv.x + dv.y * dv.y + dv.z * dv.z);
            }

            // Convert thermal kinetic energy into temperature.
            //
            // Formula used:
            //   T_cell = [sum_i m_i |v_i - u|^2] / [3 * k_B * N]
            //
            // where:
            // - u   is the mass-weighted mean velocity in the cell
            // - k_B is the Boltzmann constant
            // - N   is the number of valid particles in the cell
            field[cell] = thermal_energy_sum
                / (static_cast<T>(3)
                   * static_cast<T>(atlas::boltzmann_constant)
                   * static_cast<T>(count));
        });

    // If the selected mode requires particle-side temperature updates,
    // sample the computed cell temperature back into each particle.
    if (update_fluid) {
        // Precompute the valid upper grid index used for clamping.
        const Vector3<int> hi = grid_size - Vector3<int> { 1, 1, 1 };

        // Precompute the valid lower grid index used for clamping.
        const Vector3<int> lo { 0, 0, 0 };

        // Process all particles independently.
        //
        // For each particle:
        // 1. map world position into cell-space coordinates
        // 2. floor to integer grid coordinates
        // 3. clamp to valid domain bounds
        // 4. flatten the 3D cell index into a linear index
        // 5. copy the corresponding cell temperature into the particle
        atlas::parallel_for<ExecutionPolicy::device>(
            0,
            particle_count,
            [=] ATLAS_DEVICE(const int i) {
                // Convert particle position from world space into relative cell-space coordinates.
                const Vector3<T> rel = (position[i] - lower_corner) * inv_h;

                // Convert continuous cell-space coordinates into integer grid coordinates.
                Vector3<int> ijk = atlas::math::floor(rel).template cast_to<int>();

                // Clamp the grid coordinates so particles slightly outside the
                // domain still map to the nearest valid cell.
                ijk = atlas::math::clamp(ijk, lo, hi);

                // Flatten the 3D grid coordinate into the field array index.
                const int cell = ijk.x + ijk.y * grid_size.x + ijk.z * grid_size.x * grid_size.y;

                // Assign the sampled cell temperature to the particle.
                particle_temperature[i] = field[cell];
            });
    }
}

template <typename T>
ThermometerOperator<T>::ThermometerOperator() noexcept
    : type(ThermometerType::Variance) {
    // Default-construct the thermometer operator as the Variance variant.
    //
    // Design choice:
    // - this tagged union must always have exactly one active member
    // - the default state is therefore initialized to the Variance operator
    //
    // Active union member after construction:
    // - variance
    new (&variance) VarianceThermometerOperator<T> {};
}

template <typename T>
ThermometerOperator<T>::ThermometerOperator(const ThermometerType type_) noexcept
    : type(type_) {
    // Construct the tagged thermometer operator from an explicit type tag.
    //
    // Responsibilities:
    // - inspect the requested runtime thermometer type
    // - placement-new the matching union member
    // - report invalid tags through logging
    //
    // Important note:
    // - unlike fallback-based implementations, this version does not recover to
    //   another valid variant when the tag is invalid
    // - therefore an invalid tag leaves the object in a logically inconsistent state
    //   because no valid union member is constructed before returning
    switch (type) {
    case ThermometerType::Variance:
        // Activate the Variance thermometer operator in-place.
        new (&variance) VarianceThermometerOperator<T> {};
        return;

    default:
        // Log unexpected thermometer type values.
        //
        // This branch indicates that the runtime tag is unsupported by the
        // current ThermometerOperator<T> implementation.
        atlas::logger::error() << "Invalid ThermometerType: " << static_cast<int>(type);
        return;
    }
}

template <typename T>
ThermometerOperator<T>::ThermometerOperator(const ThermometerOperator& other) noexcept
    : type(other.type) {
    // Copy-construct from another tagged thermometer operator.
    //
    // Steps:
    // 1. copy the active runtime tag
    // 2. reconstruct the matching active union member from the source object
    copy_from(other);
}

template <typename T>
ThermometerOperator<T>&
ThermometerOperator<T>::operator=(const ThermometerOperator& other) noexcept {
    // Copy-assign from another tagged thermometer operator.
    //
    // Steps:
    // 1. detect self-assignment
    // 2. destroy the currently active union member
    // 3. copy the incoming runtime tag
    // 4. reconstruct the incoming active union member
    if (this == &other) return *this;

    // Destroy the currently active variant before switching to the new one.
    destroy_active();

    // Copy the incoming runtime tag so dispatch and reconstruction follow
    // the source object's active thermometer type.
    type = other.type;

    // Reconstruct the active union member from the source object.
    copy_from(other);

    return *this;
}

template <typename T>
ThermometerOperator<T>::~ThermometerOperator() noexcept {
    // Destroy whichever union member is currently active according to `type`.
    destroy_active();
}

template <typename T>
ThermometerOperator<T>::ThermometerOperator(const VarianceThermometerOperator<T>& op) noexcept
    : type(ThermometerType::Variance) {
    // Construct the tagged thermometer operator directly from a concrete
    // VarianceThermometerOperator<T>.
    //
    // Active union member after construction:
    // - variance
    new (&variance) VarianceThermometerOperator<T>(op);
}

template <typename T>
void
ThermometerOperator<T>::measure(const DomainDeviceProbe<T>& domain,
                                const SpatialHashingProbe<T>& searcher,
                                const FluidDeviceProbe<T>& particle,
                                const MeasureModeType measure_mode) const {
    // Dispatch the measurement request to the active thermometer variant.
    //
    // Inputs forwarded unchanged:
    // - domain       : domain-side device probe
    // - searcher     : spatial hashing / cell lookup probe
    // - particle     : particle-side device probe
    // - measure_mode : measurement direction / policy
    switch (type) {
    case ThermometerType::Variance:
        // Forward to the active Variance thermometer implementation.
        variance.measure(domain, searcher, particle, measure_mode);
        return;

    default:
        // Log unsupported runtime thermometer tags instead of dispatching.
        atlas::logger::error() << "Invalid ThermometerType: " << static_cast<int>(type);
        return;
    }
}

template <typename T>
void
ThermometerOperator<T>::destroy_active() noexcept {
    // Destroy the union member selected by the current runtime tag.
    //
    // This function assumes that `type` correctly describes the active member.
    switch (type) {
    case ThermometerType::Variance:
        // Destroy the active Variance thermometer operator.
        variance.~VarianceThermometerOperator<T>();
        return;

    default:
        // Log unexpected runtime tags.
        //
        // In this path, no union member is destroyed because the implementation
        // does not know which member is actually active.
        atlas::logger::error() << "Invalid ThermometerType: " << static_cast<int>(type);
        return;
    }
}

template <typename T>
void
ThermometerOperator<T>::copy_from(const ThermometerOperator& other) noexcept {
    // Reconstruct the active union member from another tagged operator.
    //
    // Precondition:
    // - `type` already contains the tag of the member that should be created
    //
    // This function only placement-news the selected member.
    // It does not destroy any previously active member.
    switch (type) {
    case ThermometerType::Variance:
        // Copy-construct the Variance operator from the source object's
        // corresponding active union member.
        new (&variance) VarianceThermometerOperator<T>(other.variance);
        return;

    default:
        // Log unsupported runtime tags.
        //
        // In this branch, no union member is constructed, which means the object
        // remains without a valid active member for this tag.
        atlas::logger::error() << "Invalid ThermometerType: " << static_cast<int>(type);
        return;
    }
}

} // namespace atlas::system