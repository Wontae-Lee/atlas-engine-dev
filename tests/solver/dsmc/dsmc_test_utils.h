#pragma once

#include "../../utilities/test_utils.h"

#include <atlas/fluid/fluid.h>
#include <atlas/fluid/fluid_state.h>
#include <atlas/generator/generator.h>
#include <atlas/material/material_properties.h>
#include <atlas/searcher/spatial_hashing_searcher.h>
#include <atlas/universe/universe.h>

#include <cstddef>

namespace atlas::test {

inline atlas::MaterialProperties<float>
make_dsmc_material() {
    return atlas::MaterialProperties<float>::builder()
        .with_type(atlas::MaterialType::Molecule)
        .with_mass(1.0f)
        .with_molecular_mass(1.0f)
        .with_reference_diameter(1.0f)
        .with_reference_temperature(1.0f)
        .with_viscosity_index(0.75f)
        .with_scattering_parameter(1.0f)
        .build();
}

inline atlas::UniverseHostPtr<float>
make_dsmc_universe() {
    return atlas::Universe<float>::builder()
        .with_lower_corner(atlas::Vector3F(-1.0f, -1.0f, -1.0f))
        .with_upper_corner(atlas::Vector3F(1.0f, 1.0f, 1.0f))
        .with_cell_size(1.0f)
        .make_host_shared();
}

inline atlas::FluidHostPtr<float>
make_dsmc_fluid(const std::size_t buffer_size = 4) {
    atlas::HostBuffer<atlas::MaterialProperties<float>> properties(1);
    properties[0] = make_dsmc_material();

    atlas::HostBuffer<atlas::GeneratorHostPtr<float>> generators(1);

    return atlas::Fluid<float>::builder()
        .with_properties(properties)
        .with_generators(generators)
        .with_buffer_size(buffer_size)
        .make_host_shared();
}

inline atlas::SearcherHostPtr<float>
make_dsmc_searcher(const atlas::UniverseHostPtr<float>& universe,
                   const atlas::FluidHostPtr<float>& fluid) {
    return atlas::SpatialHashingSearcher<float>::builder()
        .with_universe(universe)
        .with_fluid(fluid)
        .make_host_shared();
}

} // namespace atlas::test
