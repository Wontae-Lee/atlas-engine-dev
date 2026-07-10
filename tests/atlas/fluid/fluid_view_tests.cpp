#include <atlas/fluid/fluid.h>
#include <atlas/fluid/fluid_state.h>
#include <atlas/fluid/fluid_view.h>
#include <atlas/math/math.h>

#include <gtest/gtest.h>

#include <cstddef>
#include <type_traits>

namespace {

using atlas::Fluid;
using atlas::FluidDsmcView;
using atlas::FluidSpeciesState;
using atlas::FluidVelocityState;
using atlas::tol;

}

TEST(FluidDsmcView, IsTriviallyCopyable) {
    // The view must be capturable by value into a __host__ __device__ lambda.
    EXPECT_TRUE(std::is_trivially_copyable_v<FluidDsmcView>);
}

TEST(FluidDsmcView, DefaultConstructedHasNullPointers) {
    const FluidDsmcView view {};
    EXPECT_EQ(view.velocity, nullptr);
    EXPECT_EQ(view.species, nullptr);
    EXPECT_FALSE(view.is_complete());
}

TEST(FluidDsmcView, MakeGathersColumnsAndScalars) {
    Fluid fluid = Fluid::builder()
                      .with_buffer_size(16)
                      .with_particle_count(9)
                      .with_statistical_weight(3.5f)
                      .build();

    const FluidDsmcView view = FluidDsmcView::make(fluid);
    EXPECT_NE(view.velocity, nullptr);
    EXPECT_NE(view.species, nullptr);
    EXPECT_TRUE(view.is_complete());
    EXPECT_EQ(view.particle_count, 9);
    EXPECT_NEAR(view.statistical_weight, 3.5f, tol);
}

TEST(FluidDsmcView, MemberViewDelegatesToMake) {
    Fluid fluid = Fluid::builder().with_buffer_size(8).with_particle_count(4).build();

    const FluidDsmcView view = fluid.view<FluidDsmcView>();
    EXPECT_NE(view.velocity, nullptr);
    EXPECT_NE(view.species, nullptr);
    EXPECT_TRUE(view.is_complete());
    EXPECT_EQ(view.particle_count, 4);
}

TEST(FluidDsmcView, IncompleteWhenVelocityColumnMissing) {
    Fluid fluid(8);
    static_cast<void>(fluid.remove_state<FluidVelocityState>());

    const FluidDsmcView view = FluidDsmcView::make(fluid);
    EXPECT_EQ(view.velocity, nullptr);
    EXPECT_NE(view.species, nullptr);
    EXPECT_FALSE(view.is_complete());
}

TEST(FluidDsmcView, IncompleteWhenSpeciesColumnMissing) {
    Fluid fluid(8);
    static_cast<void>(fluid.remove_state<FluidSpeciesState>());

    const FluidDsmcView view = FluidDsmcView::make(fluid);
    EXPECT_NE(view.velocity, nullptr);
    EXPECT_EQ(view.species, nullptr);
    EXPECT_FALSE(view.is_complete());
}

TEST(FluidDsmcView, DefaultFluidYieldsNullPointers) {
    // A default fluid registers no columns, so both required pointers stay null.
    Fluid fluid {};

    const FluidDsmcView view = FluidDsmcView::make(fluid);
    EXPECT_EQ(view.velocity, nullptr);
    EXPECT_EQ(view.species, nullptr);
    EXPECT_FALSE(view.is_complete());
    EXPECT_EQ(view.particle_count, 0);
}
