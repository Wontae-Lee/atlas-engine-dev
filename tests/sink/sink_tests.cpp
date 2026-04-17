#include "../utilities/tests_utils.h"

#include <atlas/fluid/fluid.h>
#include <atlas/generator/generate_operator.h>
#include <atlas/geometry/box.h>
#include <atlas/sink/sink.h>
#include <atlas/sync/sync.h>
#include <atlas/unit/unit.h>

#include <gtest/gtest.h>

namespace {

using T = float;
using Vec3 = atlas::Vector3<T>;

atlas::FluidHostPtr<T>
make_fluid() {
    return atlas::fluid::Fluid<T>::builder()
        .with_buffer_size(8)
        .make_host_shared();
}

atlas::Unit<T>
make_unit() {
    const auto geometry = atlas::geometry::Box<T>::builder()
                              .with_lower_corner(Vec3(-1, -1, -1))
                              .with_upper_corner(Vec3(1, 1, 1))
                              .make_host_shared();

    const auto sync = atlas::physics::Sync<T>::builder()
                          .make_host_shared();

    return atlas::physics::Unit<T>::builder()
        .with_geometry(geometry)
        .with_sync(sync)
        .build();
}

atlas::fluid::DespawnOperator<T>
make_despawn_operator() {
    return atlas::fluid::DespawnOperator<T>(atlas::fluid::DespawnType::Surface);
}

} // namespace

TEST(Sink, BuilderConstructsUsableSink) {
    const auto fluid = make_fluid();

    auto sink = atlas::fluid::Sink<T>::builder()
                    .with_units(atlas::HostBuffer<atlas::Unit<T>> { make_unit() })
                    .with_fluid(fluid)
                    .with_despawn_types(atlas::HostBuffer<atlas::fluid::DespawnType> { atlas::fluid::DespawnType::Surface })
                    .with_despawn_operator(make_despawn_operator())
                    .with_tolerance(0.1f)
                    .with_flip(true)
                    .build();

    EXPECT_NO_THROW(sink.update(0.1f));
    EXPECT_NO_THROW(sink.sink());
}

TEST(Sink, BuilderRejectsMissingDependencies) {
    const auto fluid = make_fluid();

    EXPECT_THROW(
        atlas::fluid::Sink<T>::builder()
            .with_fluid(fluid)
            .with_despawn_types(atlas::HostBuffer<atlas::fluid::DespawnType> { atlas::fluid::DespawnType::Surface })
            .with_despawn_operator(make_despawn_operator())
            .build(),
        std::runtime_error);

    EXPECT_THROW(
        atlas::fluid::Sink<T>::builder()
            .with_units(atlas::HostBuffer<atlas::Unit<T>> { make_unit() })
            .with_despawn_types(atlas::HostBuffer<atlas::fluid::DespawnType> { atlas::fluid::DespawnType::Surface })
            .with_despawn_operator(make_despawn_operator())
            .build(),
        std::runtime_error);
}

TEST(Sink, BuilderRejectsMismatchedDespawnConfigurationSizes) {
    const auto fluid = make_fluid();

    EXPECT_THROW(
        atlas::fluid::Sink<T>::builder()
            .with_units(atlas::HostBuffer<atlas::Unit<T>> { make_unit(), make_unit() })
            .with_fluid(fluid)
            .with_despawn_types(atlas::HostBuffer<atlas::fluid::DespawnType> { atlas::fluid::DespawnType::Surface })
            .with_despawn_operators(atlas::HostBuffer<atlas::fluid::DespawnOperator<T>> { make_despawn_operator(), make_despawn_operator(), make_despawn_operator() })
            .build(),
        std::runtime_error);

    EXPECT_THROW(
        atlas::fluid::Sink<T>::builder()
            .with_units(atlas::HostBuffer<atlas::Unit<T>> { make_unit(), make_unit() })
            .with_fluid(fluid)
            .with_despawn_types(atlas::HostBuffer<atlas::fluid::DespawnType> { atlas::fluid::DespawnType::Surface, atlas::fluid::DespawnType::Volume, atlas::fluid::DespawnType::Surface })
            .with_despawn_operator(make_despawn_operator())
            .build(),
        std::runtime_error);
}

TEST(Sink, BuilderRejectsInvalidImmediateInputs) {
    EXPECT_THROW(
        atlas::fluid::Sink<T>::builder()
            .with_units(atlas::HostBuffer<atlas::Unit<T>> {}),
        std::runtime_error);

    EXPECT_THROW(
        atlas::fluid::Sink<T>::builder()
            .with_despawn_types(atlas::HostBuffer<atlas::fluid::DespawnType> {}),
        std::runtime_error);

    EXPECT_THROW(
        atlas::fluid::Sink<T>::builder()
            .with_despawn_operators(atlas::HostBuffer<atlas::fluid::DespawnOperator<T>> {}),
        std::runtime_error);
}

TEST(Sink, MakeHostSharedBuildsSink) {
    const auto fluid = make_fluid();

    const auto sink = atlas::fluid::Sink<T>::builder()
                          .with_units(atlas::HostBuffer<atlas::Unit<T>> { make_unit() })
                          .with_fluid(fluid)
                          .with_despawn_types(atlas::HostBuffer<atlas::fluid::DespawnType> { atlas::fluid::DespawnType::Surface })
                          .with_despawn_operator(make_despawn_operator())
                          .make_host_shared();

    ASSERT_NE(sink, nullptr);
    EXPECT_NO_THROW(sink->update(0.1f));
}

TEST(Sink, UpdateIgnoresNonPositiveDt) {
    const auto fluid = make_fluid();

    auto sink = atlas::fluid::Sink<T>::builder()
                    .with_units(atlas::HostBuffer<atlas::Unit<T>> { make_unit() })
                    .with_fluid(fluid)
                    .with_despawn_types(atlas::HostBuffer<atlas::fluid::DespawnType> { atlas::fluid::DespawnType::Surface })
                    .with_despawn_operator(make_despawn_operator())
                    .build();

    EXPECT_NO_THROW(sink.update(0.0f));
}
