#pragma once

namespace atlas::system {

template <typename T>
void
VarianceThermometerOperator<T>::measure(const DomainDeviceProbe<T>& domain,
                                        const SpatialHashingProbe<T>& searcher,
                                        const FluidDeviceProbe<T>& particle,
                                        const MeasureModeType measure_mode) const {

    // Cache the total number of cells in the domain.
    //
    // This value defines:
    // - how many field-temperature slots exist
    // - how many independent cell-wise variance-temperature computations
    //   will be launched below
    const auto num_of_cells = domain.num_of_cells;

    // ---------------------------------------------------------------------
    // Pass 1:
    // Compute one temperature value per cell from the velocity variance of
    // the particles assigned to that cell.
    //
    // Conceptually, for each cell we compute:
    //
    //   1) mass-weighted mean velocity
    //        u = [sum_i m_i v_i] / [sum_i m_i]
    //
    //   2) thermal fluctuation energy
    //        E_th = sum_i m_i |v_i - u|^2
    //
    //   3) temperature estimate
    //        T_cell = E_th / (3 k_B N)
    //
    // where:
    // - m_i is the mass of particle i
    // - v_i is the velocity of particle i
    // - N   is the number of valid particles in the cell
    // - k_B is the Boltzmann constant
    //
    // Each cell is processed independently, so this maps naturally to a
    // device-side parallel_for over the cell index range.
    // ---------------------------------------------------------------------
    atlas::parallel_for<ExecutionPolicy::device>(
        0,
        num_of_cells,
        [=] ATLAS_DEVICE(const int cell) {
            // Read the first slot in the searcher's sorted particle-index array
            // that belongs to this cell.
            //
            // Interpretation:
            // - searcher.cell_start[cell] gives the inclusive begin position
            //   of the particle range for this cell
            // - a negative value typically means the cell is empty
            const int begin = searcher.cell_start[cell];

            // Read the one-past-the-end slot of this cell's particle range.
            //
            // The valid entries for this cell therefore lie in:
            //   [begin, end)
            const int end = searcher.cell_end[cell];

            // Accumulate the numerator of the mass-weighted mean velocity:
            //
            //   sum_i (m_i * v_i)
            //
            // This is stored as a full 3D vector because velocity is vector-valued.
            Vector3<T> mean_velocity { T(0), T(0), T(0) };

            // Accumulate the denominator of the mass-weighted mean velocity:
            //
            //   sum_i m_i
            //
            // This must be strictly positive before dividing.
            T momentum_weight_sum = T(0);

            // Count the number of valid particles that actually contribute to
            // the cell statistics.
            //
            // This count is used later in the temperature normalization.
            int count = 0;

            // -------------------------------------------------------------
            // First pass over all particles assigned to this cell.
            //
            // Goals of this pass:
            // - validate indices
            // - read each particle's species
            // - read the corresponding mass
            // - accumulate mass-weighted velocity
            // - accumulate total mass
            // - count contributors
            // -------------------------------------------------------------
            for (int k = begin; k < end; ++k) {

                // Recover the original particle index from the searcher's
                // cell-sorted particle index array.
                //
                // The searcher stores particle indices indirectly so that
                // particles can be grouped by cell without rearranging the
                // particle storage itself.
                const int particle_index = searcher.indices[k];

                // Defensively skip invalid indices.
                //
                // This protects against:
                // - negative sentinel values
                // - stale indices beyond the current active particle count
                if (particle_index < 0 || particle_index >= particle.particle_count) continue;

                // Read the species id of the current particle.
                //
                // Each particle refers to one species entry in the particle
                // property table.
                const size_t species_index = particle.species[particle_index];

                // Read the particle mass from the species property table.
                //
                // Variance temperature uses mass-weighted statistics, so mass
                // is required for both the mean velocity and fluctuation energy.
                const T mass = particle.particle_property[species_index].mass;

                // Accumulate mass-weighted velocity contribution:
                //
                //   mean_velocity_numerator += m_i * v_i
                mean_velocity += particle.vel[particle_index] * mass;

                // Accumulate total mass:
                //
                //   momentum_weight_sum += m_i
                momentum_weight_sum += mass;

                // Record one additional valid contributing particle.
                ++count;
            }

            // If there are no valid particles in the cell, or if total mass is
            // not positive, then no meaningful variance temperature can be formed.
            //
            // In that case, define the field temperature for this cell as zero.
            if (count <= 0 || !(momentum_weight_sum > T(0))) {
                domain.field_temperature[cell] = T(0);
                return;
            }

            // Finish the mass-weighted mean velocity computation:
            //
            //   u = [sum_i m_i v_i] / [sum_i m_i]
            //
            // After this line, mean_velocity stores the cell's bulk velocity.
            mean_velocity /= momentum_weight_sum;

            // Accumulate the thermal fluctuation energy relative to the local
            // mean velocity:
            //
            //   sum_i m_i |v_i - u|^2
            //
            // This quantity measures the random kinetic motion around the bulk
            // flow, which is the part interpreted as thermal energy.
            T thermal_energy_sum = T(0);

            // -------------------------------------------------------------
            // Second pass over the particles in this cell.
            //
            // Goals of this pass:
            // - compute fluctuating velocity relative to the cell mean
            // - accumulate mass-weighted squared fluctuation
            // -------------------------------------------------------------
            for (int k = begin; k < end; ++k) {

                // Recover the original particle index from the searcher's
                // cell-sorted index array.
                const int particle_index = searcher.indices[k];

                // Again, skip invalid or stale indices defensively.
                if (particle_index < 0 || particle_index >= particle.particle_count) continue;

                // Read the particle's species id.
                const size_t species_index = particle.species[particle_index];

                // Read the corresponding species mass.
                const T mass = particle.particle_property[species_index].mass;

                // Compute the particle's fluctuating velocity relative to the
                // cell's mass-weighted mean velocity:
                //
                //   dv = v_i - u
                const Vector3<T> dv = particle.vel[particle_index] - mean_velocity;

                // Accumulate the mass-weighted squared fluctuation:
                //
                //   thermal_energy_sum += m_i * (dv_x^2 + dv_y^2 + dv_z^2)
                //
                // This corresponds to the total translational thermal energy-like
                // contribution in the cell.
                thermal_energy_sum += mass * (dv.x * dv.x + dv.y * dv.y + dv.z * dv.z);
            }

            // Convert the accumulated fluctuation energy into temperature.
            //
            // Formula used:
            //
            //   T_cell = [sum_i m_i |v_i - u|^2] / [3 * k_B * N]
            //
            // Interpretation of factors:
            // - numerator   : total mass-weighted velocity fluctuation energy
            // - 3           : three translational degrees of freedom
            // - k_B         : Boltzmann constant
            // - N           : number of contributing particles
            //
            // The result is stored directly into the domain field-temperature array.
            domain.field_temperature[cell] = thermal_energy_sum
                / (static_cast<T>(3)
                   * static_cast<T>(atlas::boltzmann_constant)
                   * static_cast<T>(count));
        });

    // ---------------------------------------------------------------------
    // Pass 2:
    // If the selected mode requests particle-side temperature updates,
    // propagate each cell's computed field temperature back to the particles
    // belonging to that cell.
    //
    // Current behavior:
    // - this branch executes only for MeasureModeType::Fluid
    // - it does not execute for MeasureModeType::All in this implementation
    //
    // Important indexing note:
    // - the loop variable k iterates over entries in searcher.indices
    // - the correct particle slot is searcher.indices[k]
    // - the code below writes particle.temperature[k], which assumes that
    //   the searcher ordering and particle storage ordering coincide
    // ---------------------------------------------------------------------
    if (measure_mode == MeasureModeType::Fluid || measure_mode == MeasureModeType::All) {

        // Iterate over all cells again so each cell can broadcast its already
        // computed field temperature to the particles assigned to it.
        atlas::parallel_for<ExecutionPolicy::device>(
            0,
            num_of_cells,
            [=] ATLAS_DEVICE(const int cell) {
                // Read the first slot in the searcher's sorted particle-index
                // array that belongs to this cell.
                const int begin = searcher.cell_start[cell];

                // Read the one-past-the-end slot for this cell.
                const int end = searcher.cell_end[cell];

                // Iterate over all searcher entries belonging to the current cell.
                for (int k = begin; k < end; ++k) {

                    // Assign the cell temperature to the particle-temperature array.
                    //
                    // Intended meaning:
                    // - every particle in this cell receives the same temperature,
                    //   namely the variance-derived temperature of the cell
                    const int particle_index             = searcher.indices[k];
                    particle.temperature[particle_index] = domain.field_temperature[cell];
                }
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