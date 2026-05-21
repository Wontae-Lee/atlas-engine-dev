#include "../utilities/test_utils.h"

#include <atlas/codec/codec.h>
#include <atlas/fluid/fluid.h>
#include <atlas/measure/measurer.h>
#include <atlas/orchestrator/orchestrator.h>
#include <atlas/searcher/spatial_hashing_searcher.h>
#include <atlas/universe/universe.h>

#include <testkit/testkit.h>

namespace {

using atlas::boltzmann_constant;
using atlas::Codec;
using atlas::DeviceBuffer;
using atlas::Fluid;
using atlas::FluidHostPtr;
using atlas::GeneratorHostPtr;
using atlas::HostBuffer;
using atlas::make_host_shared;
using atlas::MaterialProperties;
using atlas::MaterialType;
using atlas::MeasureModeType;
using atlas::Measurer;
using atlas::Orchestrator;
using atlas::SpatialHashingSearcher;
using atlas::SpatialHashingSearcherHostPtr;
using atlas::tol;
using atlas::Universe;
using atlas::UniverseHostPtr;
using atlas::Vector3F;
using atlas::fluid::FluidActiveState;
using atlas::fluid::FluidPositionState;
using atlas::fluid::FluidSpeciesState;
using atlas::fluid::FluidVelocityState;
using atlas::system::Solver;
using atlas::test::vec_near;
using atlas::universe::UniverseFieldForceState;
using atlas::universe::UniverseGravityState;

class MockCodec final : public Codec<float> {
public:
    MockCodec(UniverseHostPtr<float> universe,
              FluidHostPtr<float> fluid,
              SpatialHashingSearcherHostPtr<float> searcher)
        : Codec<float>(std::move(universe), std::move(fluid), std::move(searcher)) { }

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

class MockMeasurer final : public Measurer<float> {
public:
    MockMeasurer(UniverseHostPtr<float> universe,
                 FluidHostPtr<float> fluid,
                 SpatialHashingSearcherHostPtr<float> searcher)
        : Measurer<float>(std::move(universe), std::move(fluid), std::move(searcher)) { }

    void
    measure() override {
        ++measure_calls;
    }

    ATLAS_NODISCARD MeasureModeType
    measure_mode() const noexcept override {
        return MeasureModeType::Field;
    }

    int measure_calls { 0 };
};

class MockSolver final : public Solver<float> {
public:
    void
    solve(const float dt) override {
        ++plain_calls;
        last_dt = dt;
    }

    void
    solve(const DeviceBuffer<int>* allocated_solver, const int index, const float dt) override {
        ++codec_calls;
        last_dt    = dt;
        last_index = index;
        if (allocated_solver != nullptr && !allocated_solver->empty()) {
            first_allocated_value = (*allocated_solver)[0];
        }
    }

    int plain_calls { 0 };
    int codec_calls { 0 };
    int last_index { -1 };
    int first_allocated_value { -1 };
    float last_dt { 0 };
};

FluidHostPtr<float>
make_fluid() {
    HostBuffer<MaterialProperties<float>> properties(1);
    HostBuffer<GeneratorHostPtr<float>> generators(1);

    properties[0].type           = MaterialType::Molecule;
    properties[0].mass           = 2.0f;
    properties[0].molecular_mass = 2.0f;
    properties[0].species_id     = 0;

    return Fluid<float>::builder()
        .with_buffer_size(8)
        .with_statistical_weight(1.0f)
        .with_properties(properties)
        .with_generators(generators)
        .make_host_shared();
}

UniverseHostPtr<float>
make_universe() {
    return Universe<float>::builder()
        .with_lower_corner(Vector3F(0, 0, 0))
        .with_upper_corner(Vector3F(1, 1, 1))
        .with_cell_size(1.0f)
        .make_host_shared();
}

SpatialHashingSearcherHostPtr<float>
make_searcher(const UniverseHostPtr<float>& universe,
              const FluidHostPtr<float>& fluid) {
    return SpatialHashingSearcher<float>::builder()
        .with_universe(universe)
        .with_fluid(fluid)
        .make_host_shared();
}

void
seed_one_particle(const FluidHostPtr<float>& fluid, const Vector3F& position, const Vector3F& velocity) {
    auto* position_state = fluid->state<FluidPositionState<float>>();
    auto* velocity_state = fluid->state<FluidVelocityState<float>>();
    auto* species_state  = fluid->state<FluidSpeciesState<float>>();
    auto* active_state   = fluid->state<FluidActiveState<float>>();

    ASSERT_NE(position_state, nullptr);
    ASSERT_NE(velocity_state, nullptr);
    ASSERT_NE(species_state, nullptr);
    ASSERT_NE(active_state, nullptr);

    position_state->data()[0] = position;
    velocity_state->data()[0] = velocity;
    species_state->data()[0]  = 0u;
    active_state->data()[0]   = 1;
    fluid->set_particle_count(1);
}

} // namespace

TEST(Orchestrator, BuilderRejectsNullSolver) {
    EXPECT_THROW(
        Orchestrator<float>::builder()
            .with_solver(nullptr)
            .build(),
        std::runtime_error);
}

TEST(Orchestrator, BuilderWithGravityCreatesGravityState) {
    const auto universe = make_universe();

    const auto orchestrator = Orchestrator<float>::builder()
                                  .with_universe(universe)
                                  .with_gravity(Vector3F(0.0f, -9.81f, 1.25f))
                                  .build();

    (void)orchestrator;

    const auto* gravity_state = universe->state<UniverseGravityState<float>>();
    ASSERT_NE(gravity_state, nullptr);
    ASSERT_EQ(gravity_state->size(), static_cast<std::size_t>(universe->number_of_cells()));
    EXPECT_TRUE(vec_near(gravity_state->data()[0], Vector3F(0.0f, -9.81f, 1.25f), tol));
}

TEST(Orchestrator, UpdateWithoutCodecUsesPlainSolverAndMeasurer) {
    const auto fluid    = make_fluid();
    const auto universe = make_universe();
    const auto searcher = make_searcher(universe, fluid);

    auto solver   = make_host_shared<MockSolver>();
    auto measurer = make_host_shared<MockMeasurer>(universe, fluid, searcher);

    auto orchestrator = Orchestrator<float>::builder()
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
    const auto fluid    = make_fluid();
    const auto universe = make_universe();
    const auto searcher = make_searcher(universe, fluid);

    auto solver0  = make_host_shared<MockSolver>();
    auto solver1  = make_host_shared<MockSolver>();
    auto codec    = make_host_shared<MockCodec>(universe, fluid, searcher);
    auto measurer = make_host_shared<MockMeasurer>(universe, fluid, searcher);

    auto orchestrator = Orchestrator<float>::builder()
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
    const auto fluid    = make_fluid();
    const auto universe = make_universe();
    const auto searcher = make_searcher(universe, fluid);

    seed_one_particle(fluid, Vector3F(0.25f, 0.25f, 0.25f), Vector3F(1.0f, 2.0f, 3.0f));
    searcher->build();

    auto orchestrator = Orchestrator<float>::builder()
                            .with_universe(universe)
                            .with_fluid(fluid)
                            .with_searcher(searcher)
                            .with_gravity(Vector3F(0.0f, -9.0f, 2.0f))
                            .build();

    orchestrator.update(0.5f);

    const auto* velocity_state = fluid->state<FluidVelocityState<float>>();
    ASSERT_NE(velocity_state, nullptr);
    EXPECT_TRUE(vec_near(
        velocity_state->data()[0],
        Vector3F(1.0f, -2.5f, 4.0f),
        tol));
}

TEST(Orchestrator, ApplyFieldForceUpdatesParticleVelocityUsingMass) {
    const auto fluid    = make_fluid();
    const auto universe = make_universe();
    const auto searcher = make_searcher(universe, fluid);

    seed_one_particle(fluid, Vector3F(0.25f, 0.25f, 0.25f), Vector3F(0.0f, 0.0f, 0.0f));
    searcher->build();

    universe->set_state<UniverseFieldForceState<float>>(
        std::make_unique<UniverseFieldForceState<float>>(
            DeviceBuffer<Vector3F>(static_cast<std::size_t>(universe->number_of_cells()), Vector3F(2.0f, 0.0f, 0.0f))));

    auto orchestrator = Orchestrator<float>::builder()
                            .with_universe(universe)
                            .with_fluid(fluid)
                            .with_searcher(searcher)
                            .build();

    orchestrator.update(0.5f);

    const auto* velocity_state = fluid->state<FluidVelocityState<float>>();
    ASSERT_NE(velocity_state, nullptr);
    EXPECT_TRUE(vec_near(
        velocity_state->data()[0],
        Vector3F(0.5f, 0.0f, 0.0f),
        tol));
}
