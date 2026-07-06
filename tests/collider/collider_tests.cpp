#include <atlas/collider/collider.h>

#include <atlas/geometry/geometry.h>
#include <atlas/geometry/box.h>
#include <atlas/geometry/plane.h>
#include <atlas/memory/copy.h>
#include <atlas/memory/raw_pointer_cast.h>
#include <atlas/sync/sync.h>
#include <atlas/unit/unit.h>
#include <atlas/universe/universe.h>

#include <gtest/gtest.h>

#include <cstdint>
#include <stdexcept>

namespace {

using atlas::Box;
using atlas::Collider;
using atlas::Fluid;
using atlas::FluidHostPtr;
using atlas::FluidInternalEnergy;
using atlas::FluidInternalEnergyState;
using atlas::FluidPositionState;
using atlas::FluidSpeciesState;
using atlas::FluidVelocityState;
using atlas::HostBuffer;
using atlas::IsothermalSurfaceInteraction;
using atlas::MaterialProperties;
using atlas::MaxwellianSurfaceInteraction;
using atlas::Plane;
using atlas::Sync;
using atlas::Unit;
using atlas::Universe;
using atlas::UniverseHostPtr;
using atlas::Vector3;
using atlas::tol;

void
expect_vec_near(const Vector3& actual, const Vector3& expected) {
    EXPECT_NEAR(actual.x, expected.x, tol);
    EXPECT_NEAR(actual.y, expected.y, tol);
    EXPECT_NEAR(actual.z, expected.z, tol);
}

FluidHostPtr
make_fluid() {
    return Fluid::builder()
        .with_buffer_size(8)
        .make_host_shared();
}

UniverseHostPtr
make_universe(const HostBuffer<Unit>& collider_units) {
    return Universe::builder()
        .with_lower_corner(Vector3(-10.0f, -10.0f, -10.0f))
        .with_upper_corner(Vector3(10.0f, 10.0f, 10.0f))
        .with_cell_size(1.0f)
        .with_collider_units(collider_units)
        .make_host_shared();
}

Unit
make_unit() {
    static const auto geometry = Box::builder()
                                     .with_lower_corner(Vector3(-1.0f, -1.0f, -1.0f))
                                     .with_upper_corner(Vector3(1.0f, 1.0f, 1.0f))
                                     .make_host_shared();

    const auto sync = Sync::builder()
                          .make_host_shared();

    return Unit::builder()
        .with_geometry(atlas::Geometry(*geometry))
        .with_sync(sync)
        .build();
}

Unit
make_plane_unit(const Vector3& linear_velocity  = Vector3(0.0f, 0.0f, 0.0f),
                const Vector3& angular_velocity = Vector3(0.0f, 0.0f, 0.0f)) {
    static const auto geometry = Plane::builder()
                                     .with_point_normal(Vector3(0.0f, 0.0f, 0.0f), Vector3(1.0f, 0.0f, 0.0f))
                                     .make_host_shared();

    const auto sync = Sync::builder()
                          .make_host_shared();

    auto builder = Unit::builder()
                       .with_geometry(atlas::Geometry(*geometry))
                       .with_sync(sync);

    if (linear_velocity.length_squared() > 0.0f) {
        builder.with_velocity(linear_velocity);
    }

    if (angular_velocity.length_squared() > 0.0f) {
        builder.with_angular_velocity(angular_velocity);
    }

    return builder.build();
}

IsothermalSurfaceInteraction
make_interaction() {
    return IsothermalSurfaceInteraction::builder()
        .with_restitution(0.9f)
        .with_momentum_acc(0.2f)
        .build();
}

IsothermalSurfaceInteraction
make_specular_interaction() {
    return IsothermalSurfaceInteraction::builder()
        .with_restitution(1.0f)
        .with_momentum_acc(0.0f)
        .build();
}

}

TEST(Collider, EmptyReflectsMissingDependencies) {
    const Collider empty_collider;

    EXPECT_TRUE(empty_collider.empty());
}

TEST(Collider, BuilderConstructsUsableCollider) {
    const auto fluid = make_fluid();

    const auto collider = Collider::builder()
                              .with_universe(make_universe(HostBuffer<Unit> { make_unit() }))
                              .with_fluid(fluid)
                              .with_surface_interactions(
                                  HostBuffer<IsothermalSurfaceInteraction> { make_interaction() })
                              .build();

    EXPECT_FALSE(collider.empty());
}

TEST(Collider, ConstructorAcceptsPreparedBuffers) {
    const auto fluid = make_fluid();
    const HostBuffer<Unit> units { make_unit() };
    const HostBuffer<atlas::SurfaceInteractionKernel> interactions {
        atlas::SurfaceInteractionKernel(make_interaction())
    };
    const HostBuffer<std::uint8_t> flips { std::uint8_t { 0 } };

    const Collider collider(
        make_universe(units),
        atlas::DeviceBuffer<atlas::SurfaceInteractionKernel>(interactions.begin(), interactions.end()),
        atlas::DeviceBuffer<std::uint8_t>(flips.begin(), flips.end()),
        atlas::PostColliderType::fast,
        fluid);

    EXPECT_FALSE(collider.empty());
}

TEST(Collider, BuilderUsesDefaultIsothermalFastConfiguration) {
    const auto fluid = make_fluid();

    const auto collider = Collider::builder()
                              .with_universe(make_universe(HostBuffer<Unit> { make_unit() }))
                              .with_fluid(fluid)
                              .build();

    EXPECT_FALSE(collider.empty());
    EXPECT_NO_THROW(collider.collide(0.01f));
}

TEST(Collider, MakeProbeReflectsConfiguredRuntimeState) {
    const auto fluid = make_fluid();
    fluid->set_particle_count(1);

    const auto collider = Collider::builder()
                              .with_universe(make_universe(HostBuffer<Unit> { make_unit() }))
                              .with_fluid(fluid)
                              .with_surface_interactions(
                                  HostBuffer<IsothermalSurfaceInteraction> { make_interaction() })
                              .build();

    EXPECT_TRUE(collider.make_probe());
}

TEST(Collider, BuilderRejectsInvalidConfiguration) {
    const auto fluid = make_fluid();

    EXPECT_THROW(
        static_cast<void>(Collider::builder()
                              .with_fluid(fluid)
                              .build()),
        std::runtime_error);

    EXPECT_THROW(
        static_cast<void>(Collider::builder()
                              .with_universe(make_universe(HostBuffer<Unit> { make_unit() }))
                              .build()),
        std::runtime_error);

    EXPECT_THROW(
        static_cast<void>(Collider::builder()
                              .with_universe(make_universe(HostBuffer<Unit> {}))
                              .with_fluid(fluid)
                              .build()),
        std::runtime_error);

    EXPECT_THROW(
        static_cast<void>(Collider::builder()
                              .with_universe(make_universe(HostBuffer<Unit> { make_unit(), make_unit() }))
                              .with_fluid(fluid)
                              .with_surface_interactions(
                                  HostBuffer<IsothermalSurfaceInteraction> { make_interaction(), make_interaction(), make_interaction() })
                              .build()),
        std::runtime_error);

    EXPECT_THROW(
        static_cast<void>(Collider::builder()
                              .with_universe(make_universe(HostBuffer<Unit> { make_unit(), make_unit() }))
                              .with_fluid(fluid)
                              .with_surface_interaction_kernels(HostBuffer<atlas::SurfaceInteractionKernel> {})
                              .build()),
        std::runtime_error);

    EXPECT_THROW(
        static_cast<void>(Collider::builder()
                              .with_universe(make_universe(HostBuffer<Unit> { make_unit() }))
                              .with_fluid(fluid)
                              .with_surface_interactions(HostBuffer<IsothermalSurfaceInteraction> {})
                              .build()),
        std::runtime_error);

    EXPECT_THROW(
        static_cast<void>(Collider::builder()
                              .with_universe(make_universe(HostBuffer<Unit> { make_unit() }))
                              .with_fluid(fluid)
                              .with_surface_interactions(HostBuffer<MaxwellianSurfaceInteraction> {})
                              .build()),
        std::runtime_error);

    EXPECT_THROW(
        static_cast<void>(Collider::builder()
                              .with_universe(make_universe(HostBuffer<Unit> { make_unit() }))
                              .with_fluid(fluid)
                              .with_flips(HostBuffer<std::uint8_t> {})
                              .build()),
        std::runtime_error);

    EXPECT_THROW(
        static_cast<void>(Collider::builder()
                              .with_universe(make_universe(HostBuffer<Unit> { make_unit(), make_unit() }))
                              .with_fluid(fluid)
                              .with_flips(HostBuffer<std::uint8_t> { 0, 0, 0 })
                              .build()),
        std::runtime_error);
}

TEST(Collider, MakeHostSharedBuildsCollider) {
    const auto fluid = make_fluid();

    const auto collider = Collider::builder()
                              .with_universe(make_universe(HostBuffer<Unit> { make_unit() }))
                              .with_fluid(fluid)
                              .with_surface_interactions(
                                  HostBuffer<IsothermalSurfaceInteraction> { make_interaction() })
                              .make_host_shared();

    ASSERT_NE(collider, nullptr);
    EXPECT_FALSE(collider->empty());
}

TEST(Collider, BuilderAcceptsTaggedSurfaceInteractionAndFlipConfigurations) {
    const auto fluid       = make_fluid();
    const auto interaction = atlas::SurfaceInteractionKernel(make_interaction());

    const auto shared_interaction = Collider::builder()
                                        .with_universe(make_universe(HostBuffer<Unit> { make_unit() }))
                                        .with_fluid(fluid)
                                        .with_surface_interaction_kernel(interaction)
                                        .with_flip(true)
                                        .build();
    EXPECT_FALSE(shared_interaction.empty());

    const auto per_unit = Collider::builder()
                              .with_universe(make_universe(HostBuffer<Unit> { make_unit(), make_unit() }))
                              .with_fluid(fluid)
                              .with_surface_interaction_kernels(
                                  HostBuffer<atlas::SurfaceInteractionKernel> { interaction, interaction })
                              .with_flips(HostBuffer<std::uint8_t> { 0, 1 })
                              .build();
    EXPECT_FALSE(per_unit.empty());
}

TEST(Collider, UpdateAndCollideAreSafeNoOpsForDefaultFluidState) {
    const auto fluid = make_fluid();

    auto collider = Collider::builder()
                        .with_universe(make_universe(HostBuffer<Unit> { make_unit() }))
                        .with_fluid(fluid)
                        .with_surface_interactions(
                            HostBuffer<IsothermalSurfaceInteraction> { make_interaction() })
                        .build();

    EXPECT_NO_THROW(collider.update(0.01f));
    EXPECT_NO_THROW(collider.collide(0.01f));
}

TEST(Collider, CollideAccountsForColliderLinearVelocityInSurfaceResponse) {
    const auto fluid = make_fluid();
    fluid->set_particle_count(1);

    auto* positions  = fluid->state<FluidPositionState>();
    auto* velocities = fluid->state<FluidVelocityState>();
    ASSERT_NE(positions, nullptr);
    ASSERT_NE(velocities, nullptr);

    positions->data()[0]  = Vector3(-1.0f, 0.0f, 0.0f);
    velocities->data()[0] = Vector3(1.0f, 0.0f, 0.0f);

    auto collider = Collider::builder()
                        .with_universe(make_universe(HostBuffer<Unit> {
                            make_plane_unit(Vector3(0.0f, 2.0f, 0.0f))
                        }))
                        .with_fluid(fluid)
                        .with_surface_interactions(
                            HostBuffer<IsothermalSurfaceInteraction> { make_specular_interaction() })
                        .build();

    collider.collide(1.0f);

    expect_vec_near(
        fluid->state<FluidVelocityState>()->data()[0],
        Vector3(-1.0f, 0.0f, 0.0f));
}

TEST(Collider, CollideAccountsForColliderAngularVelocityAtContactPoint) {
    const auto fluid = make_fluid();
    fluid->set_particle_count(1);

    auto* positions  = fluid->state<FluidPositionState>();
    auto* velocities = fluid->state<FluidVelocityState>();
    ASSERT_NE(positions, nullptr);
    ASSERT_NE(velocities, nullptr);

    positions->data()[0]  = Vector3(-1.0f, 1.0f, 0.0f);
    velocities->data()[0] = Vector3(1.0f, 0.0f, 0.0f);

    auto collider = Collider::builder()
                        .with_universe(make_universe(HostBuffer<Unit> {
                            make_plane_unit(Vector3(0.0f, 0.0f, 0.0f), Vector3(0.0f, 0.0f, 1.0f))
                        }))
                        .with_fluid(fluid)
                        .with_surface_interactions(
                            HostBuffer<IsothermalSurfaceInteraction> { make_specular_interaction() })
                        .build();

    collider.collide(1.0f);

    expect_vec_near(
        fluid->state<FluidVelocityState>()->data()[0],
        Vector3(-3.0f, 0.0f, 0.0f));
}

TEST(Collider, CollideUpdatesInternalEnergyThroughSurfaceInteraction) {
    const auto fluid = make_fluid();
    fluid->set_particle_count(1);

    auto* positions         = fluid->state<FluidPositionState>();
    auto* velocities        = fluid->state<FluidVelocityState>();
    auto* species           = fluid->state<FluidSpeciesState>();
    auto& internal_energies = fluid->emplace_state<FluidInternalEnergyState>(fluid->buffer_size());
    ASSERT_NE(positions, nullptr);
    ASSERT_NE(velocities, nullptr);
    ASSERT_NE(species, nullptr);

    positions->data()[0]        = Vector3(-1.0f, 0.0f, 0.0f);
    velocities->data()[0]       = Vector3(1.0f, 0.0f, 0.0f);
    species->data()[0]          = 0u;
    internal_energies.data()[0] = FluidInternalEnergy { 1.0f, 2.0f, 3.0f };
    const auto material = MaterialProperties::builder()
                              .with_mass(1.0f)
                              .with_molecular_mass(1.0f)
                              .with_rotational_dof(2)
                              .with_vibrational_dof(2)
                              .build();
    fluid->particle_properties().push_back(material);

    const auto interaction = MaxwellianSurfaceInteraction::builder()
                                 .with_temperature(300.0f)
                                 .with_molecular_mass(2.0f)
                                 .with_accommodation(1.0f, 1.0f, 1.0f, 1.0f)
                                 .build();
    const auto expected = interaction.internal_energy(
        FluidInternalEnergy { 1.0f, 2.0f, 3.0f },
        Vector3(1.0f, 0.0f, 0.0f),
        Vector3(1.0f, 0.0f, 0.0f),
        material);

    auto collider = Collider::builder()
                        .with_universe(make_universe(HostBuffer<Unit> { make_plane_unit() }))
                        .with_fluid(fluid)
                        .with_surface_interactions(
                            HostBuffer<MaxwellianSurfaceInteraction> { interaction })
                        .build();

    collider.collide(1.0f);

    FluidInternalEnergy actual {};
    atlas::copy_device_to_host(atlas::raw_pointer_cast(internal_energies.data().data()), &actual, 1);
    EXPECT_NEAR(actual.translational, expected.translational, tol);
    EXPECT_NEAR(actual.rotational, expected.rotational, tol);
    EXPECT_NEAR(actual.vibrational, expected.vibrational, tol);
}

TEST(Collider, CollidePreservesInternalEnergyForIsothermalSurfaceInteraction) {
    const auto fluid = make_fluid();
    fluid->set_particle_count(1);

    auto* positions         = fluid->state<FluidPositionState>();
    auto* velocities        = fluid->state<FluidVelocityState>();
    auto* species           = fluid->state<FluidSpeciesState>();
    auto& internal_energies = fluid->emplace_state<FluidInternalEnergyState>(fluid->buffer_size());
    ASSERT_NE(positions, nullptr);
    ASSERT_NE(velocities, nullptr);
    ASSERT_NE(species, nullptr);

    const FluidInternalEnergy incident_energy { 1.0f, 2.0f, 3.0f };
    positions->data()[0]        = Vector3(-1.0f, 0.0f, 0.0f);
    velocities->data()[0]       = Vector3(1.0f, 0.0f, 0.0f);
    species->data()[0]          = 0u;
    internal_energies.data()[0] = incident_energy;
    fluid->particle_properties().push_back(
        MaterialProperties::builder()
            .with_mass(1.0f)
            .with_molecular_mass(1.0f)
            .build());

    auto collider = Collider::builder()
                        .with_universe(make_universe(HostBuffer<Unit> { make_plane_unit() }))
                        .with_fluid(fluid)
                        .with_surface_interactions(
                            HostBuffer<IsothermalSurfaceInteraction> { make_specular_interaction() })
                        .build();

    collider.collide(1.0f);

    FluidInternalEnergy actual {};
    atlas::copy_device_to_host(atlas::raw_pointer_cast(internal_energies.data().data()), &actual, 1);
    EXPECT_NEAR(actual.translational, incident_energy.translational, tol);
    EXPECT_NEAR(actual.rotational, incident_energy.rotational, tol);
    EXPECT_NEAR(actual.vibrational, incident_energy.vibrational, tol);
}
