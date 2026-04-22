#include "../../utilities/tests_utils.h"

#include <atlas/generator/generate_operator.h>
#include <atlas/material/material_properties.h>
#include <atlas/solver/sph/sph_gateway_solver.h>

#include <testkit/testkit.h>

namespace {

using T = float;
using Vec3 = atlas::Vector3<T>;

atlas::UniverseHostPtr<T>
make_universe() {
    return atlas::universe::Universe<T>::builder()
        .with_lower_corner(Vec3(0, 0, 0))
        .with_upper_corner(Vec3(1, 1, 1))
        .with_cell_size(0.5f)
        .make_host_shared();
}

atlas::FluidHostPtr<T>
make_fluid() {
    atlas::HostBuffer<atlas::MatrialProperties<T>> properties;
    properties.push_back(
        atlas::MatrialProperties<T>::builder()
            .with_type(atlas::MaterialType::Molecule)
            .with_mass(1.0f)
            .with_molecular_mass(1.0f)
            .with_rest_density(0.1f)
            .with_pressure_coefficient(4.0f)
            .with_dynamic_viscosity(0.05f)
            .with_smoothing_length(0.35f)
            .build());

    atlas::HostBuffer<atlas::GeneratorHostPtr<T>> generators;
    generators.push_back(nullptr);

    return atlas::fluid::Fluid<T>::builder()
        .with_buffer_size(8)
        .with_properties(properties)
        .with_generators(generators)
        .make_host_shared();
}

atlas::SpatialHashingSearcherHostPtr<T>
make_searcher(const atlas::UniverseHostPtr<T>& universe,
              const atlas::FluidHostPtr<T>& fluid) {
    return atlas::system::SpatialHashingSearcher<T>::builder()
        .with_universe(universe)
        .with_fluid(fluid)
        .make_host_shared();
}

} // namespace

TEST(SphGatewaySolver, BuilderConstructsConfiguredSolver) {
    const auto universe = make_universe();
    const auto fluid = make_fluid();
    const auto searcher = make_searcher(universe, fluid);

    const auto solver = atlas::system::SphGatewaySolver<T>::builder()
                            .with_universe(universe)
                            .with_fluid(fluid)
                            .with_searcher(searcher)
                            .with_kernel_type(atlas::system::SphKernelType::wendland_quintic)
                            .with_group_particle_count(3)
                            .build();

    EXPECT_EQ(solver.kernel_type(), atlas::system::SphKernelType::wendland_quintic);
    EXPECT_EQ(solver.group_particle_count(), 3);
}

TEST(SphGatewaySolver, CodecAwareSolveSharesStateWithinEachDeterministicGroup) {
    const auto universe = make_universe();
    const auto fluid = make_fluid();
    const auto searcher = make_searcher(universe, fluid);

    fluid->set_particle_count(4);
    fluid->state<atlas::fluid::FluidSpeciesState<T>>()->data()[0] = 0u;
    fluid->state<atlas::fluid::FluidSpeciesState<T>>()->data()[1] = 0u;
    fluid->state<atlas::fluid::FluidSpeciesState<T>>()->data()[2] = 0u;
    fluid->state<atlas::fluid::FluidSpeciesState<T>>()->data()[3] = 0u;

    fluid->state<atlas::fluid::FluidPositionState<T>>()->data()[0] = Vec3(0.10f, 0.10f, 0.10f);
    fluid->state<atlas::fluid::FluidPositionState<T>>()->data()[1] = Vec3(0.12f, 0.10f, 0.10f);
    fluid->state<atlas::fluid::FluidPositionState<T>>()->data()[2] = Vec3(0.30f, 0.10f, 0.10f);
    fluid->state<atlas::fluid::FluidPositionState<T>>()->data()[3] = Vec3(0.32f, 0.10f, 0.10f);

    fluid->state<atlas::fluid::FluidVelocityState<T>>()->data()[0] = Vec3(1.0f, 0.0f, 0.0f);
    fluid->state<atlas::fluid::FluidVelocityState<T>>()->data()[1] = Vec3(3.0f, 0.0f, 0.0f);
    fluid->state<atlas::fluid::FluidVelocityState<T>>()->data()[2] = Vec3(5.0f, 0.0f, 0.0f);
    fluid->state<atlas::fluid::FluidVelocityState<T>>()->data()[3] = Vec3(7.0f, 0.0f, 0.0f);

    atlas::system::SphGatewaySolver<T> solver(
        universe,
        fluid,
        searcher,
        atlas::system::SphKernelType::standard,
        2);

    atlas::DeviceBuffer<int> allocated_solver(static_cast<std::size_t>(universe->number_of_cells()), -1);
    allocated_solver[0] = 0;

    solver.solve(&allocated_solver, 0, 0.01f);

    const auto p0 = fluid->state<atlas::fluid::FluidPositionState<T>>()->data()[0];
    const auto p1 = fluid->state<atlas::fluid::FluidPositionState<T>>()->data()[1];
    const auto p2 = fluid->state<atlas::fluid::FluidPositionState<T>>()->data()[2];
    const auto p3 = fluid->state<atlas::fluid::FluidPositionState<T>>()->data()[3];

    const auto v0 = fluid->state<atlas::fluid::FluidVelocityState<T>>()->data()[0];
    const auto v1 = fluid->state<atlas::fluid::FluidVelocityState<T>>()->data()[1];
    const auto v2 = fluid->state<atlas::fluid::FluidVelocityState<T>>()->data()[2];
    const auto v3 = fluid->state<atlas::fluid::FluidVelocityState<T>>()->data()[3];

    EXPECT_TRUE(atlas::test::vec_near(p0, p1, static_cast<T>(1e-5)));
    EXPECT_TRUE(atlas::test::vec_near(p2, p3, static_cast<T>(1e-5)));
    EXPECT_TRUE(atlas::test::vec_near(v0, v1, static_cast<T>(1e-5)));
    EXPECT_TRUE(atlas::test::vec_near(v2, v3, static_cast<T>(1e-5)));

    EXPECT_GT(universe->state<atlas::universe::UniverseNumberParticleState<T>>()->data()[0], 0.0f);
}
