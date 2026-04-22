#include "../utilities/tests_utils.h"

#include <atlas/codec/codec.h>
#include <atlas/fluid/fluid.h>
#include <atlas/measure/measurer.h>
#include <atlas/orchestrator/orchestrator.h>
#include <atlas/searcher/spatial_hashing_searcher.h>
#include <atlas/universe/universe.h>

#include <testkit/testkit.h>

namespace {

using T = float;
using Vec3 = atlas::Vector3<T>;

class MockCodec final : public atlas::system::Codec<T> {
public:
    MockCodec(atlas::UniverseHostPtr<T> universe,
              atlas::FluidHostPtr<T> fluid,
              atlas::SpatialHashingSearcherHostPtr<T> searcher)
        : atlas::system::Codec<T>(std::move(universe), std::move(fluid), std::move(searcher)) { }

    void
    encode() override {
        ++encode_calls;
    }

    void
    decode() override {
        ++decode_calls;

        for (std::size_t i = 0; i < this->allocated_solver().size(); ++i) {
            this->allocated_solver()[i] = static_cast<int>(i % 2);
        }
    }

    int encode_calls { 0 };
    int decode_calls { 0 };
};

class MockMeasurer final : public atlas::system::Measurer<T> {
public:
    MockMeasurer(atlas::UniverseHostPtr<T> universe,
                 atlas::FluidHostPtr<T> fluid,
                 atlas::SpatialHashingSearcherHostPtr<T> searcher)
        : atlas::system::Measurer<T>(std::move(universe), std::move(fluid), std::move(searcher)) { }

    void
    measure() override {
        ++measure_calls;
    }

    atlas::MeasureModeType
    measure_mode() const noexcept override {
        return atlas::MeasureModeType::Field;
    }

    int measure_calls { 0 };
};

class MockSolver final : public atlas::system::Solver<T> {
public:
    void
    solve(const T dt) override {
        ++plain_calls;
        last_dt = dt;
    }

    void
    solve(const atlas::DeviceBuffer<int>* allocated_solver, const int index, const T dt) override {
        ++codec_calls;
        last_dt = dt;
        last_index = index;
        if (allocated_solver != nullptr && !allocated_solver->empty()) {
            first_allocated_value = (*allocated_solver)[0];
        }
    }

    int plain_calls { 0 };
    int codec_calls { 0 };
    int last_index { -1 };
    int first_allocated_value { -1 };
    T last_dt { 0 };
};

atlas::FluidHostPtr<T>
make_fluid() {
    atlas::HostBuffer<atlas::system::MaterialProperties<T>> properties(1);
    atlas::HostBuffer<atlas::GeneratorHostPtr<T>> generators(1);

    properties[0].type = atlas::system::MaterialType::Molecule;
    properties[0].mass = 2.0f;
    properties[0].molecular_mass = 2.0f;
    properties[0].species_id = 0;

    return atlas::fluid::Fluid<T>::builder()
        .with_buffer_size(8)
        .with_statistical_weight(1.0f)
        .with_properties(properties)
        .with_generators(generators)
        .make_host_shared();
}

atlas::UniverseHostPtr<T>
make_universe() {
    return atlas::universe::Universe<T>::builder()
        .with_lower_corner(Vec3(0, 0, 0))
        .with_upper_corner(Vec3(1, 1, 1))
        .with_cell_size(1.0f)
        .make_host_shared();
}

atlas::SpatialHashingSearcherHostPtr<T>
make_searcher(const atlas::UniverseHostPtr<T>& universe,
              const atlas::FluidHostPtr<T>& fluid) {
    return atlas::SpatialHashingSearcher<T>::builder()
        .with_universe(universe)
        .with_fluid(fluid)
        .make_host_shared();
}

void
seed_one_particle(const atlas::FluidHostPtr<T>& fluid, const Vec3& position, const Vec3& velocity) {
    auto* position_state = fluid->state<atlas::fluid::FluidPositionState<T>>();
    auto* velocity_state = fluid->state<atlas::fluid::FluidVelocityState<T>>();
    auto* species_state = fluid->state<atlas::fluid::FluidSpeciesState<T>>();
    auto* active_state = fluid->state<atlas::fluid::FluidActiveState<T>>();

    ASSERT_NE(position_state, nullptr);
    ASSERT_NE(velocity_state, nullptr);
    ASSERT_NE(species_state, nullptr);
    ASSERT_NE(active_state, nullptr);

    position_state->data()[0] = position;
    velocity_state->data()[0] = velocity;
    species_state->data()[0] = 0u;
    active_state->data()[0] = 1;
    fluid->set_particle_count(1);
}

} // namespace

TEST(Orchestrator, BuilderRejectsNullSolver) {
    EXPECT_THROW(
        atlas::Orchestrator<T>::builder()
            .with_solver(nullptr)
            .build(),
        std::runtime_error);
}

TEST(Orchestrator, BuilderWithGravityCreatesGravityState) {
    const auto universe = make_universe();

    const auto orchestrator = atlas::Orchestrator<T>::builder()
                                  .with_universe(universe)
                                  .with_gravity(Vec3(0.0f, -9.81f, 1.25f))
                                  .build();

    (void)orchestrator;

    const auto* gravity_state = universe->state<atlas::universe::UniverseGravityState<T>>();
    ASSERT_NE(gravity_state, nullptr);
    ASSERT_EQ(gravity_state->size(), static_cast<std::size_t>(universe->number_of_cells()));
    EXPECT_TRUE(atlas::test::vec_near(gravity_state->data()[0], Vec3(0.0f, -9.81f, 1.25f), static_cast<T>(1e-5)));
}

TEST(Orchestrator, UpdateWithoutCodecUsesPlainSolverAndMeasurer) {
    const auto fluid = make_fluid();
    const auto universe = make_universe();
    const auto searcher = make_searcher(universe, fluid);

    auto solver = atlas::make_host_shared<MockSolver>();
    auto measurer = atlas::make_host_shared<MockMeasurer>(universe, fluid, searcher);

    auto orchestrator = atlas::Orchestrator<T>::builder()
                            .with_universe(universe)
                            .with_fluid(fluid)
                            .with_searcher(searcher)
                            .with_measurer(measurer)
                            .with_solver(solver)
                            .make_host_shared();

    orchestrator->update(0.25f);

    EXPECT_EQ(measurer->measure_calls, 1);
    EXPECT_EQ(solver->plain_calls, 1);
    EXPECT_EQ(solver->codec_calls, 0);
    EXPECT_FLOAT_EQ(solver->last_dt, 0.25f);
}

TEST(Orchestrator, UpdateWithCodecUsesCodecAwareSolve) {
    const auto fluid = make_fluid();
    const auto universe = make_universe();
    const auto searcher = make_searcher(universe, fluid);

    auto solver0 = atlas::make_host_shared<MockSolver>();
    auto solver1 = atlas::make_host_shared<MockSolver>();
    auto codec = atlas::make_host_shared<MockCodec>(universe, fluid, searcher);
    auto measurer = atlas::make_host_shared<MockMeasurer>(universe, fluid, searcher);

    auto orchestrator = atlas::Orchestrator<T>::builder()
                            .with_universe(universe)
                            .with_fluid(fluid)
                            .with_searcher(searcher)
                            .with_codec(codec)
                            .with_measurer(measurer)
                            .with_solver(solver0)
                            .with_solver(solver1)
                            .make_host_shared();

    orchestrator->update(0.5f);

    EXPECT_EQ(codec->encode_calls, 1);
    EXPECT_EQ(codec->decode_calls, 1);
    EXPECT_EQ(measurer->measure_calls, 1);

    EXPECT_EQ(solver0->plain_calls, 0);
    EXPECT_EQ(solver0->codec_calls, 1);
    EXPECT_EQ(solver0->last_index, 0);
    EXPECT_EQ(solver0->first_allocated_value, 0);
    EXPECT_FLOAT_EQ(solver0->last_dt, 0.5f);

    EXPECT_EQ(solver1->plain_calls, 0);
    EXPECT_EQ(solver1->codec_calls, 1);
    EXPECT_EQ(solver1->last_index, 1);
    EXPECT_EQ(solver1->first_allocated_value, 0);
    EXPECT_FLOAT_EQ(solver1->last_dt, 0.5f);
}

TEST(Orchestrator, ApplyGravityUpdatesParticleVelocity) {
    const auto fluid = make_fluid();
    const auto universe = make_universe();
    const auto searcher = make_searcher(universe, fluid);

    seed_one_particle(fluid, Vec3(0.25f, 0.25f, 0.25f), Vec3(1.0f, 2.0f, 3.0f));
    searcher->build();

    auto orchestrator = atlas::Orchestrator<T>::builder()
                            .with_universe(universe)
                            .with_fluid(fluid)
                            .with_searcher(searcher)
                            .with_gravity(Vec3(0.0f, -9.0f, 2.0f))
                            .build();

    orchestrator.apply_gravity(0.5f);

    const auto* velocity_state = fluid->state<atlas::fluid::FluidVelocityState<T>>();
    ASSERT_NE(velocity_state, nullptr);
    EXPECT_TRUE(atlas::test::vec_near(
        velocity_state->data()[0],
        Vec3(1.0f, -2.5f, 4.0f),
        static_cast<T>(1e-5)));
}

TEST(Orchestrator, ApplyFieldForceUpdatesParticleVelocityUsingMass) {
    const auto fluid = make_fluid();
    const auto universe = make_universe();
    const auto searcher = make_searcher(universe, fluid);

    seed_one_particle(fluid, Vec3(0.25f, 0.25f, 0.25f), Vec3(0.0f, 0.0f, 0.0f));
    searcher->build();

    universe->set_state<atlas::universe::UniverseFieldForceState<T>>(
        std::make_unique<atlas::universe::UniverseFieldForceState<T>>(
            atlas::DeviceBuffer<Vec3>(static_cast<std::size_t>(universe->number_of_cells()), Vec3(2.0f, 0.0f, 0.0f))));

    auto orchestrator = atlas::Orchestrator<T>::builder()
                            .with_universe(universe)
                            .with_fluid(fluid)
                            .with_searcher(searcher)
                            .build();

    orchestrator.apply_field_force(0.5f);

    const auto* velocity_state = fluid->state<atlas::fluid::FluidVelocityState<T>>();
    ASSERT_NE(velocity_state, nullptr);
    EXPECT_TRUE(atlas::test::vec_near(
        velocity_state->data()[0],
        Vec3(0.5f, 0.0f, 0.0f),
        static_cast<T>(1e-5)));
}
