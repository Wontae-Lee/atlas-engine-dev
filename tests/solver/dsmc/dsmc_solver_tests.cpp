#include "../../utilities/tests_utils.h"

#include <atlas/generator/generate_operator.h>
#include <atlas/material/material_properties.h>
#include <atlas/solver/dsmc/dsmc_solver.h>

#include <testkit/testkit.h>

namespace {

using T = float;

atlas::UniverseHostPtr<T>
make_universe() {
    return atlas::universe::Universe<T>::builder()
        .with_lower_corner(atlas::Vector3<T>(0, 0, 0))
        .with_upper_corner(atlas::Vector3<T>(1, 1, 1))
        .with_cell_size(1.0f)
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
            .with_collision_diameter(1.0f)
            .build());

    atlas::HostBuffer<atlas::GeneratorHostPtr<T>> generators;
    generators.push_back(nullptr);

    return atlas::fluid::Fluid<T>::builder()
        .with_buffer_size(4)
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

class DummyDsmcSolver final : public atlas::system::DsmcSolver<T> {
public:
    using atlas::system::DsmcSolver<T>::DsmcSolver;

protected:
    void
    apply_collisions(const atlas::DeviceBuffer<int>*, int, T) override { }
};

} // namespace

TEST(DsmcSolver, BaseConstructorCreatesRequiredUniverseStates) {
    const auto universe = make_universe();
    const auto fluid = make_fluid();
    const auto searcher = make_searcher(universe, fluid);

    const DummyDsmcSolver solver(universe, fluid, searcher, atlas::system::DsmcKernelType::variable_soft_sphere);

    EXPECT_EQ(solver.kernel_type(), atlas::system::DsmcKernelType::variable_soft_sphere);
    ASSERT_TRUE(universe->has_state<atlas::universe::UniverseNumberParticleState<T>>());
    ASSERT_TRUE(universe->has_state<atlas::universe::UniverseMaxRelativeSpeedState<T>>());
    ASSERT_TRUE(universe->has_state<atlas::universe::UniverseCollisionCountState<int>>());
}
