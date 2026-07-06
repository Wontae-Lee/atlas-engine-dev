#pragma once

#include <atlas/fluid/fluid.h>
#include <atlas/fluid/fluid_state.h>
#include <atlas/generator/generator.h>
#include <atlas/material/material_properties.h>
#include <atlas/searcher/spatial_hashing_searcher.h>
#include <atlas/universe/universe.h>

#include <cstddef>

namespace atlas::test {

inline atlas::MaterialProperties
make_dsmc_material() {
    return atlas::MaterialProperties::builder()
        .with_type(atlas::MaterialType::molecule)
        .with_mass(1.0f)
        .with_molecular_mass(1.0f)
        .with_reference_diameter(1.0f)
        .with_reference_temperature(1.0f)
        .with_viscosity_index(0.75f)
        .with_scattering_parameter(1.0f)
        .build();
}

inline atlas::UniverseHostPtr
make_dsmc_universe() {
    return atlas::Universe::builder()
        .with_lower_corner(atlas::Float3(-1.0f, -1.0f, -1.0f))
        .with_upper_corner(atlas::Float3(1.0f, 1.0f, 1.0f))
        .with_cell_size(1.0f)
        .make_host_shared();
}

inline atlas::FluidHostPtr
make_dsmc_fluid(const std::size_t buffer_size = 4) {
    atlas::HostBuffer<atlas::MaterialProperties> properties(1);
    properties[0] = make_dsmc_material();

    atlas::HostBuffer<atlas::GeneratorHostPtr> generators(1);

    return atlas::Fluid::builder()
        .with_properties(properties)
        .with_generators(generators)
        .with_buffer_size(buffer_size)
        .make_host_shared();
}

inline atlas::SearcherHostPtr
make_dsmc_searcher(const atlas::UniverseHostPtr& universe,
                   const atlas::FluidHostPtr& fluid) {
    return atlas::SpatialHashingSearcher::builder()
        .with_universe(universe)
        .with_fluid(fluid)
        .make_host_shared();
}

}
