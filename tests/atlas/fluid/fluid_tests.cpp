#include <atlas/fluid/fluid.h>
#include <atlas/fluid/fluid_state.h>
#include <atlas/material/material.h>
#include <atlas/material/material_dictionary.h>
#include <atlas/material/molecule.h>
#include <atlas/math/math.h>

#include <gtest/gtest.h>

#include <cstddef>
#include <memory>
#include <stdexcept>
#include <type_traits>
#include <utility>

namespace {

using atlas::Fluid;
using atlas::FluidPositionState;
using atlas::FluidSpeciesState;
using atlas::FluidTemperatureState;
using atlas::FluidVelocityState;
using atlas::Float3;
using atlas::Material;
using atlas::MaterialDictionary;
using atlas::MaterialDictionaryHostPtr;
using atlas::Molecule;
using atlas::tol;

/** Builds a single-species dictionary handle for the material-attachment tests. */
MaterialDictionaryHostPtr
make_dictionary() {
    return MaterialDictionary::builder()
        .with_material(Material(Molecule(2.0f, 1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 0.5f, 1.0f)))
        .make_host_shared();
}

}

TEST(Fluid, DefaultConstructedIsEmpty) {
    const Fluid fluid {};
    EXPECT_EQ(fluid.buffer_size(), std::size_t { 0 });
    EXPECT_EQ(fluid.particle_count(), std::size_t { 0 });
    EXPECT_NEAR(fluid.statistical_weight(), 1.0f, tol);
    EXPECT_FALSE(fluid.has_state<FluidPositionState>());
    EXPECT_EQ(fluid.materials(), nullptr);
}

TEST(Fluid, ConstructorAllocatesMandatoryColumns) {
    const Fluid fluid(16);
    EXPECT_EQ(fluid.buffer_size(), std::size_t { 16 });
    EXPECT_EQ(fluid.particle_count(), std::size_t { 0 });
    EXPECT_NEAR(fluid.statistical_weight(), 1.0f, tol);
    EXPECT_TRUE(fluid.has_state<FluidPositionState>());
    EXPECT_TRUE(fluid.has_state<FluidVelocityState>());
    EXPECT_TRUE(fluid.has_state<FluidSpeciesState>());
    // Optional columns are not seeded by construction.
    EXPECT_FALSE(fluid.has_state<FluidTemperatureState>());
    EXPECT_EQ(fluid.active().size(), std::size_t { 16 });
}

TEST(Fluid, MandatoryColumnsAreSizedToCapacity) {
    const Fluid fluid(12);
    ASSERT_NE(fluid.state<FluidPositionState>(), nullptr);
    ASSERT_NE(fluid.state<FluidVelocityState>(), nullptr);
    ASSERT_NE(fluid.state<FluidSpeciesState>(), nullptr);
    EXPECT_EQ(fluid.state<FluidPositionState>()->size(), std::size_t { 12 });
    EXPECT_EQ(fluid.state<FluidVelocityState>()->size(), std::size_t { 12 });
    EXPECT_EQ(fluid.state<FluidSpeciesState>()->size(), std::size_t { 12 });
}

TEST(Fluid, ConstructorWithMaterialsAttachesDictionary) {
    const Fluid fluid(8, make_dictionary());
    ASSERT_NE(fluid.materials(), nullptr);
    EXPECT_EQ(fluid.materials()->size(), std::size_t { 1 });
    EXPECT_EQ(fluid.buffer_size(), std::size_t { 8 });
}

TEST(Fluid, ConstructorWithNullMaterialsLeavesMaterialsNull) {
    // The two-argument constructor accepts a null dictionary: the mandatory columns
    // are still allocated, but materials() stays null.
    const Fluid fluid(8, MaterialDictionaryHostPtr {});
    EXPECT_EQ(fluid.buffer_size(), std::size_t { 8 });
    EXPECT_EQ(fluid.materials(), nullptr);
    EXPECT_TRUE(fluid.has_state<FluidPositionState>());
    EXPECT_TRUE(fluid.has_state<FluidVelocityState>());
    EXPECT_TRUE(fluid.has_state<FluidSpeciesState>());
}

TEST(Fluid, SetParticleCountUpdatesLiveCount) {
    Fluid fluid(16);
    fluid.set_particle_count(10);
    EXPECT_EQ(fluid.particle_count(), std::size_t { 10 });
}

TEST(Fluid, SetParticleCountRejectsOverflow) {
    Fluid fluid(4);
    EXPECT_THROW(fluid.set_particle_count(5), std::out_of_range);
}

TEST(Fluid, SetParticleCountAcceptsExactCapacity) {
    Fluid fluid(4);
    fluid.set_particle_count(4);
    EXPECT_EQ(fluid.particle_count(), std::size_t { 4 });
}

TEST(Fluid, StateAccessorReturnsNullWhenAbsent) {
    Fluid fluid(4);
    EXPECT_EQ(fluid.state<FluidTemperatureState>(), nullptr);
}

TEST(Fluid, EmplaceStateRegistersOptionalColumn) {
    Fluid fluid(6);
    auto& state = fluid.emplace_state<FluidTemperatureState>(std::size_t { 6 });
    EXPECT_EQ(state.size(), std::size_t { 6 });
    EXPECT_TRUE(fluid.has_state<FluidTemperatureState>());
    ASSERT_NE(fluid.state<FluidTemperatureState>(), nullptr);
    EXPECT_EQ(fluid.state<FluidTemperatureState>()->size(), std::size_t { 6 });
}

TEST(Fluid, SetStateReplacesColumn) {
    Fluid fluid(6);
    fluid.set_state(std::make_unique<FluidTemperatureState>(6));
    ASSERT_TRUE(fluid.has_state<FluidTemperatureState>());
    EXPECT_EQ(fluid.state<FluidTemperatureState>()->size(), std::size_t { 6 });
}

TEST(Fluid, RemoveStateDetachesColumn) {
    Fluid fluid(6);
    fluid.emplace_state<FluidTemperatureState>(std::size_t { 6 });
    ASSERT_TRUE(fluid.has_state<FluidTemperatureState>());

    auto removed = fluid.remove_state<FluidTemperatureState>();
    ASSERT_NE(removed, nullptr);
    EXPECT_EQ(removed->size(), std::size_t { 6 });
    EXPECT_FALSE(fluid.has_state<FluidTemperatureState>());
}

TEST(Fluid, RemoveAbsentStateReturnsNull) {
    Fluid fluid(6);
    EXPECT_EQ(fluid.remove_state<FluidTemperatureState>(), nullptr);
}

TEST(Fluid, StatesStoreHoldsMandatoryColumns) {
    Fluid fluid(4);
    EXPECT_EQ(fluid.states().size(), std::size_t { 3 });
}

TEST(Fluid, CompactPacksSurvivorsToFront) {
    Fluid fluid(4);
    fluid.set_particle_count(4);

    // Keep particles 0 and 2; drop 1 and 3.
    auto& active = fluid.active();
    active[0]    = 1;
    active[1]    = 0;
    active[2]    = 1;
    active[3]    = 0;

    auto* position = fluid.state<FluidPositionState>();
    ASSERT_NE(position, nullptr);
    auto& column  = position->data();
    column[0]     = Float3(0.0f, 0.0f, 0.0f);
    column[1]     = Float3(1.0f, 1.0f, 1.0f);
    column[2]     = Float3(2.0f, 2.0f, 2.0f);
    column[3]     = Float3(3.0f, 3.0f, 3.0f);

    const std::size_t kept = fluid.compact();
    EXPECT_EQ(kept, std::size_t { 2 });
    EXPECT_EQ(fluid.particle_count(), std::size_t { 2 });

    // Survivor slot 0 keeps old particle 0; slot 1 keeps old particle 2.
    const Float3 survivor0 = column[0];
    const Float3 survivor1 = column[1];
    EXPECT_NEAR(survivor0.x, 0.0f, tol);
    EXPECT_NEAR(survivor1.x, 2.0f, tol);
    EXPECT_NEAR(survivor1.y, 2.0f, tol);
    EXPECT_NEAR(survivor1.z, 2.0f, tol);
}

TEST(Fluid, CompactWithNoParticlesIsNoOp) {
    Fluid fluid(4);
    EXPECT_EQ(fluid.compact(), std::size_t { 0 });
    EXPECT_EQ(fluid.particle_count(), std::size_t { 0 });
}

TEST(Fluid, CompactWithAllSurvivorsKeepsCount) {
    Fluid fluid(3);
    fluid.set_particle_count(3);
    auto& active = fluid.active();
    active[0]    = 1;
    active[1]    = 1;
    active[2]    = 1;

    EXPECT_EQ(fluid.compact(), std::size_t { 3 });
    EXPECT_EQ(fluid.particle_count(), std::size_t { 3 });
}

TEST(Fluid, IsMoveOnly) {
    // The fluid owns device buffers, so it is move-only.
    EXPECT_FALSE(std::is_copy_constructible_v<Fluid>);
    EXPECT_FALSE(std::is_copy_assignable_v<Fluid>);
    EXPECT_TRUE(std::is_move_constructible_v<Fluid>);
    EXPECT_TRUE(std::is_move_assignable_v<Fluid>);
}

TEST(Fluid, MoveConstructionTransfersColumnsAndBookkeeping) {
    Fluid source(8, make_dictionary());
    source.set_particle_count(5);

    const Fluid moved = std::move(source);
    EXPECT_EQ(moved.buffer_size(), std::size_t { 8 });
    EXPECT_EQ(moved.particle_count(), std::size_t { 5 });
    EXPECT_TRUE(moved.has_state<FluidPositionState>());
    ASSERT_NE(moved.materials(), nullptr);
}

TEST(FluidBuilder, BuildsValidatedFluid) {
    const Fluid fluid = Fluid::builder()
                            .with_buffer_size(32)
                            .with_particle_count(20)
                            .with_statistical_weight(2.5f)
                            .with_materials(make_dictionary())
                            .build();

    EXPECT_EQ(fluid.buffer_size(), std::size_t { 32 });
    EXPECT_EQ(fluid.particle_count(), std::size_t { 20 });
    EXPECT_NEAR(fluid.statistical_weight(), 2.5f, tol);
    ASSERT_NE(fluid.materials(), nullptr);
    EXPECT_TRUE(fluid.has_state<FluidVelocityState>());
}

TEST(FluidBuilder, DefaultsStatisticalWeightToOne) {
    const Fluid fluid = Fluid::builder().with_buffer_size(4).build();
    EXPECT_NEAR(fluid.statistical_weight(), 1.0f, tol);
}

TEST(FluidBuilder, RejectsNonPositiveWeight) {
    EXPECT_THROW(
        static_cast<void>(Fluid::builder().with_buffer_size(4).with_statistical_weight(0.0f).build()),
        std::runtime_error);
    EXPECT_THROW(
        static_cast<void>(Fluid::builder().with_buffer_size(4).with_statistical_weight(-1.0f).build()),
        std::runtime_error);
}

TEST(FluidBuilder, RejectsParticleCountAboveCapacity) {
    EXPECT_THROW(
        static_cast<void>(Fluid::builder().with_buffer_size(4).with_particle_count(5).build()),
        std::runtime_error);
}

TEST(FluidBuilder, MakeHostUniqueReturnsOwnedFluid) {
    const auto fluid = Fluid::builder().with_buffer_size(10).with_particle_count(3).make_host_unique();
    ASSERT_NE(fluid, nullptr);
    EXPECT_EQ(fluid->buffer_size(), std::size_t { 10 });
    EXPECT_EQ(fluid->particle_count(), std::size_t { 3 });
}
