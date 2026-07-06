#include <atlas/orchestrator/orchestrator.h>

#include <atlas/codec/codec.h>
#include <atlas/fluid/fluid.h>
#include <atlas/measure/measurer.h>
#include <atlas/searcher/spatial_hashing_searcher.h>
#include <atlas/universe/universe.h>

#include <gtest/gtest.h>

#include <cstddef>
#include <memory>
#include <stdexcept>
#include <utility>

namespace {

using atlas::Codec;
using atlas::DeviceBuffer;
using atlas::Fluid;
using atlas::FluidActiveState;
using atlas::FluidHostPtr;
using atlas::FluidPositionState;
using atlas::FluidSpeciesState;
using atlas::FluidVelocityState;
using atlas::GeneratorHostPtr;
using atlas::HostBuffer;
using atlas::make_host_shared;
using atlas::MaterialProperties;
using atlas::MaterialType;
using atlas::MeasureModeType;
using atlas::Measurer;
using atlas::Orchestrator;
using atlas::SearcherHostPtr;
using atlas::Solver;
using atlas::SpatialHashingSearcher;
using atlas::tol;
using atlas::Universe;
using atlas::UniverseFieldForceState;
using atlas::UniverseGravityState;
using atlas::UniverseHostPtr;
using atlas::Vector3;

void
expect_vec_near(const Vector3& actual, const Vector3& expected) {
    EXPECT_NEAR(actual.x, expected.x, tol);
    EXPECT_NEAR(actual.y, expected.y, tol);
    EXPECT_NEAR(actual.z, expected.z, tol);
}

class MockCodec final : public Codec {
public:
    MockCodec(UniverseHostPtr universe,
              FluidHostPtr fluid,
              SearcherHostPtr searcher)
        : Codec(std::move(universe), std::move(fluid), std::move(searcher)) { }

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

class MockMeasurer final : public Measurer {
public:
    MockMeasurer(UniverseHostPtr universe,
                 FluidHostPtr fluid,
                 SearcherHostPtr searcher)
        : Measurer(std::move(universe), std::move(fluid), std::move(searcher)) { }

    void
    measure() override {
        ++measure_calls;
    }

    MeasureModeType
    measure_mode() const noexcept override {
        return MeasureModeType::field;
    }

    int measure_calls { 0 };
};

class MockSolver final : public Solver {
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
    float last_dt { 0.0f };
};

FluidHostPtr
make_fluid() {
    HostBuffer<MaterialProperties> properties(1);
    HostBuffer<GeneratorHostPtr> generators(1);

    properties[0] = MaterialProperties::builder()
                        .with_type(MaterialType::molecule)
                        .with_mass(2.0f)
                        .with_molecular_mass(2.0f)
                        .with_species_id(0)
                        .build();

    return Fluid::builder()
        .with_buffer_size(8)
        .with_statistical_weight(1.0f)
        .with_properties(properties)
        .with_generators(generators)
        .make_host_shared();
}

UniverseHostPtr
make_universe() {
    return Universe::builder()
        .with_lower_corner(Vector3(0.0f, 0.0f, 0.0f))
        .with_upper_corner(Vector3(1.0f, 1.0f, 1.0f))
        .with_cell_size(1.0f)
        .make_host_shared();
}

SearcherHostPtr
make_searcher(const UniverseHostPtr& universe,
              const FluidHostPtr& fluid) {
    return SpatialHashingSearcher::builder()
        .with_universe(universe)
        .with_fluid(fluid)
        .make_host_shared();
}

void
seed_one_particle(const FluidHostPtr& fluid, const Vector3& position, const Vector3& velocity) {
    auto* position_state = fluid->state<FluidPositionState>();
    auto* velocity_state = fluid->state<FluidVelocityState>();
    auto* species_state  = fluid->state<FluidSpeciesState>();
    auto* active_state   = fluid->state<FluidActiveState>();

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

}

TEST(Orchestrator, BuilderRejectsNullSolver) {
    EXPECT_THROW(
        static_cast<void>(Orchestrator::builder()
                              .with_solver(nullptr)
                              .build()),
        std::runtime_error);
}

TEST(Orchestrator, BuilderWithGravityCreatesGravityState) {
    const auto universe = make_universe();

    const auto orchestrator = Orchestrator::builder()
                                  .with_universe(universe)
                                  .with_gravity(Vector3(0.0f, -9.81f, 1.25f))
                                  .build();

    static_cast<void>(orchestrator);

    const auto* gravity_state = universe->state<UniverseGravityState>();
    ASSERT_NE(gravity_state, nullptr);
    ASSERT_EQ(gravity_state->size(), static_cast<std::size_t>(universe->cell_count()));
    expect_vec_near(gravity_state->data()[0], Vector3(0.0f, -9.81f, 1.25f));
}

TEST(Orchestrator, UpdateWithoutCodecUsesPlainSolverAndMeasurer) {
    const auto fluid    = make_fluid();
    const auto universe = make_universe();
    const auto searcher = make_searcher(universe, fluid);

    auto solver   = make_host_shared<MockSolver>();
    auto measurer = make_host_shared<MockMeasurer>(universe, fluid, searcher);

    auto orchestrator = Orchestrator::builder()
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

    auto orchestrator = Orchestrator::builder()
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

    seed_one_particle(fluid, Vector3(0.25f, 0.25f, 0.25f), Vector3(1.0f, 2.0f, 3.0f));
    searcher->build();

    auto orchestrator = Orchestrator::builder()
                            .with_universe(universe)
                            .with_fluid(fluid)
                            .with_searcher(searcher)
                            .with_gravity(Vector3(0.0f, -9.0f, 2.0f))
                            .build();

    orchestrator.update(0.5f);

    const auto* velocity_state = fluid->state<FluidVelocityState>();
    ASSERT_NE(velocity_state, nullptr);
    expect_vec_near(velocity_state->data()[0], Vector3(1.0f, -2.5f, 4.0f));
}

TEST(Orchestrator, ApplyFieldForceUpdatesParticleVelocityUsingMass) {
    const auto fluid    = make_fluid();
    const auto universe = make_universe();
    const auto searcher = make_searcher(universe, fluid);

    seed_one_particle(fluid, Vector3(0.25f, 0.25f, 0.25f), Vector3(0.0f, 0.0f, 0.0f));
    searcher->build();

    universe->set_state<UniverseFieldForceState>(
        std::make_unique<UniverseFieldForceState>(
            DeviceBuffer<Vector3>(static_cast<std::size_t>(universe->cell_count()), Vector3(2.0f, 0.0f, 0.0f))));

    auto orchestrator = Orchestrator::builder()
                            .with_universe(universe)
                            .with_fluid(fluid)
                            .with_searcher(searcher)
                            .build();

    orchestrator.update(0.5f);

    const auto* velocity_state = fluid->state<FluidVelocityState>();
    ASSERT_NE(velocity_state, nullptr);
    expect_vec_near(velocity_state->data()[0], Vector3(0.5f, 0.0f, 0.0f));
}
