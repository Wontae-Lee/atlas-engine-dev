#include "../utilities/test_utils.h"

#include <atlas/collider/collider.h>
#include <atlas/generator/generate_operator.h>
#include <atlas/geometry/box.h>
#include <atlas/geometry/plane.h>
#include <atlas/memory/copy.h>
#include <atlas/memory/raw_pointer_cast.h>
#include <atlas/sync/sync.h>
#include <atlas/unit/unit.h>

#include <testkit/testkit.h>

namespace {

using atlas::Box;
using atlas::Collider;
using atlas::FluidInternalEnergy;
using atlas::IsothermalSurfaceInteraction;
using atlas::Fluid;
using atlas::FluidHostPtr;
using atlas::HostBuffer;
using atlas::MaterialProperties;
using atlas::MaxwellianSurfaceInteraction;
using atlas::Plane;
using atlas::Sync;
using atlas::Unit;
using atlas::Vector3F;
using atlas::FluidInternalEnergyState;
using atlas::FluidPositionState;
using atlas::FluidSpeciesState;
using atlas::FluidVelocityState;
using atlas::test::vec_near;
using atlas::tol;

FluidHostPtr<float>
make_fluid() {
    return Fluid<float>::builder()
        .with_buffer_size(8)
        .make_host_shared();
}

Unit<float>
make_unit() {
    static const auto geometry = Box<float>::builder()
                                     .with_lower_corner(Vector3F(-1, -1, -1))
                                     .with_upper_corner(Vector3F(1, 1, 1))
                                     .make_host_shared();

    const auto sync = Sync<float>::builder()
                          .make_host_shared();

    return Unit<float>::builder()
        .with_geometry(geometry)
        .with_sync(sync)
        .build();
}

Unit<float>
make_plane_unit(const Vector3F& linear_velocity = Vector3F(0, 0, 0),
                const Vector3F& angular_velocity = Vector3F(0, 0, 0)) {
    static const auto geometry = Plane<float>::builder()
                                     .with_point_normal(Vector3F(0, 0, 0), Vector3F(1, 0, 0))
                                     .make_host_shared();

    const auto sync = Sync<float>::builder()
                          .make_host_shared();

    auto builder = Unit<float>::builder()
                       .with_geometry(geometry)
                       .with_sync(sync);

    if (!vec_near(linear_velocity, Vector3F(0, 0, 0), 0.0f)) {
        builder.with_velocity(linear_velocity);
    }

    if (!vec_near(angular_velocity, Vector3F(0, 0, 0), 0.0f)) {
        builder.with_angular_velocity(angular_velocity);
    }

    return builder.build();
}

IsothermalSurfaceInteraction<float>
make_interaction() {
    return IsothermalSurfaceInteraction<float>::builder()
        .with_restitution(0.9f)
        .with_momentum_acc(0.2f)
        .build();
}

IsothermalSurfaceInteraction<float>
make_specular_interaction() {
    return IsothermalSurfaceInteraction<float>::builder()
        .with_restitution(1.0f)
        .with_momentum_acc(0.0f)
        .build();
}

} // namespace

TEST(Collider, EmptyReflectsMissingDependencies) {
    const Collider<float> empty_collider;

    EXPECT_TRUE(empty_collider.empty());
}

TEST(Collider, BuilderConstructsUsableCollider) {
    const auto fluid = make_fluid();

    const auto collider = Collider<float>::builder()
                              .with_units(HostBuffer<Unit<float>> { make_unit() })
                              .with_fluid(fluid)
                              .with_surface_interactions(
                                  HostBuffer<IsothermalSurfaceInteraction<float>> { make_interaction() })
                              .build();

    EXPECT_FALSE(collider.empty());
}

TEST(Collider, ConstructorAcceptsPreparedBuffers) {
    const auto fluid = make_fluid();
    const HostBuffer<Unit<float>> units { make_unit() };
    const HostBuffer<atlas::SurfaceInteractionKernel<float>> interactions {
        atlas::SurfaceInteractionKernel<float>(make_interaction())
    };
    const HostBuffer<std::uint8_t> flips { std::uint8_t { 0 } };

    const Collider<float> collider(
        atlas::DeviceBuffer<Unit<float>>(units.begin(), units.end()),
        atlas::DeviceBuffer<atlas::SurfaceInteractionKernel<float>>(interactions.begin(), interactions.end()),
        atlas::DeviceBuffer<std::uint8_t>(flips.begin(), flips.end()),
        atlas::PostColliderType::fast,
        fluid);

    EXPECT_FALSE(collider.empty());
}

TEST(Collider, BuilderUsesDefaultIsothermalFastConfiguration) {
    const auto fluid = make_fluid();

    const auto collider = Collider<float>::builder()
                              .with_units(HostBuffer<Unit<float>> { make_unit() })
                              .with_fluid(fluid)
                              .build();

    EXPECT_FALSE(collider.empty());
    EXPECT_NO_THROW(collider.collide(0.01f));
}

TEST(Collider, MakeProbeReflectsConfiguredRuntimeState) {
    const auto fluid = make_fluid();
    fluid->set_particle_count(1);

    const auto collider = Collider<float>::builder()
                              .with_units(HostBuffer<Unit<float>> { make_unit() })
                              .with_fluid(fluid)
                              .with_surface_interactions(
                                  HostBuffer<IsothermalSurfaceInteraction<float>> { make_interaction() })
                              .build();

    EXPECT_TRUE(collider.make_probe());
}

TEST(Collider, BuilderRejectsInvalidConfiguration) {
    const auto fluid = make_fluid();

    EXPECT_THROW(
        Collider<float>::builder()
            .with_fluid(fluid)
            .build(),
        std::runtime_error);

    EXPECT_THROW(
        Collider<float>::builder()
            .with_units(HostBuffer<Unit<float>> { make_unit() })
            .build(),
        std::runtime_error);

    EXPECT_THROW(
        Collider<float>::builder()
            .with_units(HostBuffer<Unit<float>> {})
            .with_fluid(fluid),
        std::runtime_error);

    EXPECT_THROW(
        Collider<float>::builder()
            .with_units(HostBuffer<Unit<float>> { make_unit(), make_unit() })
            .with_fluid(fluid)
            .with_surface_interactions(
                HostBuffer<IsothermalSurfaceInteraction<float>> { make_interaction(), make_interaction(), make_interaction() })
            .build(),
        std::runtime_error);

    EXPECT_THROW(
        Collider<float>::builder()
            .with_units(HostBuffer<Unit<float>> { make_unit(), make_unit() })
            .with_fluid(fluid)
            .with_surface_interaction_kernels(HostBuffer<atlas::SurfaceInteractionKernel<float>> {})
            .build(),
        std::runtime_error);

    EXPECT_THROW(
        Collider<float>::builder()
            .with_units(HostBuffer<Unit<float>> { make_unit() })
            .with_fluid(fluid)
            .with_surface_interactions(HostBuffer<IsothermalSurfaceInteraction<float>> {})
            .build(),
        std::runtime_error);

    EXPECT_THROW(
        Collider<float>::builder()
            .with_units(HostBuffer<Unit<float>> { make_unit() })
            .with_fluid(fluid)
            .with_surface_interactions(HostBuffer<MaxwellianSurfaceInteraction<float>> {})
            .build(),
        std::runtime_error);

    EXPECT_THROW(
        Collider<float>::builder()
            .with_units(HostBuffer<Unit<float>> { make_unit() })
            .with_fluid(fluid)
            .with_flips(HostBuffer<std::uint8_t> {})
            .build(),
        std::runtime_error);

    EXPECT_THROW(
        Collider<float>::builder()
            .with_units(HostBuffer<Unit<float>> { make_unit(), make_unit() })
            .with_fluid(fluid)
            .with_flips(HostBuffer<std::uint8_t> { 0, 0, 0 })
            .build(),
        std::runtime_error);
}

TEST(Collider, MakeHostSharedBuildsCollider) {
    const auto fluid = make_fluid();

    const auto collider = Collider<float>::builder()
                              .with_units(HostBuffer<Unit<float>> { make_unit() })
                              .with_fluid(fluid)
                              .with_surface_interactions(
                                  HostBuffer<IsothermalSurfaceInteraction<float>> { make_interaction() })
                              .make_host_shared();

    ASSERT_NE(collider, nullptr);
    EXPECT_FALSE(collider->empty());
}

TEST(Collider, BuilderAcceptsTaggedSurfaceInteractionAndFlipConfigurations) {
    const auto fluid = make_fluid();
    const auto interaction = atlas::SurfaceInteractionKernel<float>(make_interaction());

    const auto shared_interaction = Collider<float>::builder()
                                        .with_units(HostBuffer<Unit<float>> { make_unit() })
                                        .with_fluid(fluid)
                                        .with_surface_interaction_kernel(interaction)
                                        .with_flip(true)
                                        .build();
    EXPECT_FALSE(shared_interaction.empty());

    const auto per_unit = Collider<float>::builder()
                              .with_units(HostBuffer<Unit<float>> { make_unit(), make_unit() })
                              .with_fluid(fluid)
                              .with_surface_interaction_kernels(
                                  HostBuffer<atlas::SurfaceInteractionKernel<float>> { interaction, interaction })
                              .with_flips(HostBuffer<std::uint8_t> { 0, 1 })
                              .build();
    EXPECT_FALSE(per_unit.empty());
}

TEST(Collider, UpdateAndCollideAreSafeNoOpsForDefaultFluidState) {
    const auto fluid = make_fluid();

    auto collider = Collider<float>::builder()
                        .with_units(HostBuffer<Unit<float>> { make_unit() })
                        .with_fluid(fluid)
                        .with_surface_interactions(
                            HostBuffer<IsothermalSurfaceInteraction<float>> { make_interaction() })
                        .build();

    EXPECT_NO_THROW(collider.update(0.01f));
    EXPECT_NO_THROW(collider.collide(0.01f));
}

TEST(Collider, CollideAccountsForColliderLinearVelocityInSurfaceResponse) {
    const auto fluid = make_fluid();
    fluid->set_particle_count(1);

    auto* positions  = fluid->state<FluidPositionState<float>>();
    auto* velocities = fluid->state<FluidVelocityState<float>>();
    ASSERT_NE(positions, nullptr);
    ASSERT_NE(velocities, nullptr);

    positions->data()[0]  = Vector3F(-1.0f, 0.0f, 0.0f);
    velocities->data()[0] = Vector3F(1.0f, 0.0f, 0.0f);

    auto collider = Collider<float>::builder()
                        .with_units(HostBuffer<Unit<float>> {
                            make_plane_unit(Vector3F(0.0f, 2.0f, 0.0f))
                        })
                        .with_fluid(fluid)
                        .with_surface_interactions(
                            HostBuffer<IsothermalSurfaceInteraction<float>> { make_specular_interaction() })
                        .build();

    collider.collide(1.0f);

    EXPECT_TRUE(vec_near(
        fluid->state<FluidVelocityState<float>>()->data()[0],
        Vector3F(-1.0f, 0.0f, 0.0f),
        tol));
}

TEST(Collider, CollideAccountsForColliderAngularVelocityAtContactPoint) {
    const auto fluid = make_fluid();
    fluid->set_particle_count(1);

    auto* positions  = fluid->state<FluidPositionState<float>>();
    auto* velocities = fluid->state<FluidVelocityState<float>>();
    ASSERT_NE(positions, nullptr);
    ASSERT_NE(velocities, nullptr);

    positions->data()[0]  = Vector3F(-1.0f, 1.0f, 0.0f);
    velocities->data()[0] = Vector3F(1.0f, 0.0f, 0.0f);

    auto collider = Collider<float>::builder()
                        .with_units(HostBuffer<Unit<float>> {
                            make_plane_unit(Vector3F(0.0f, 0.0f, 0.0f), Vector3F(0.0f, 0.0f, 1.0f))
                        })
                        .with_fluid(fluid)
                        .with_surface_interactions(
                            HostBuffer<IsothermalSurfaceInteraction<float>> { make_specular_interaction() })
                        .build();

    collider.collide(1.0f);

    EXPECT_TRUE(vec_near(
        fluid->state<FluidVelocityState<float>>()->data()[0],
        Vector3F(-3.0f, 0.0f, 0.0f),
        tol));
}

TEST(Collider, CollideUpdatesInternalEnergyThroughSurfaceInteraction) {
    const auto fluid = make_fluid();
    fluid->set_particle_count(1);

    auto* positions = fluid->state<FluidPositionState<float>>();
    auto* velocities = fluid->state<FluidVelocityState<float>>();
    auto* species = fluid->state<FluidSpeciesState<float>>();
    auto& internal_energies = fluid->emplace_state<FluidInternalEnergyState<float>>(fluid->buffer_size());
    ASSERT_NE(positions, nullptr);
    ASSERT_NE(velocities, nullptr);
    ASSERT_NE(species, nullptr);

    positions->data()[0] = Vector3F(-1.0f, 0.0f, 0.0f);
    velocities->data()[0] = Vector3F(1.0f, 0.0f, 0.0f);
    species->data()[0] = 0u;
    internal_energies.data()[0] = FluidInternalEnergy<float> { 1.0f, 2.0f, 3.0f };
    const auto material = MaterialProperties<float>::builder()
                              .with_mass(1.0f)
                              .with_molecular_mass(1.0f)
                              .with_rotational_dof(2)
                              .with_vibrational_dof(2)
                              .build();
    fluid->particle_properties().push_back(material);

    const auto interaction = MaxwellianSurfaceInteraction<float>::builder()
                                 .with_temperature(300.0f)
                                 .with_molecular_mass(2.0f)
                                 .with_accommodation(1.0f, 1.0f, 1.0f, 1.0f)
                                 .build();
    const auto expected = interaction.internal_energy(
        internal_energies.data()[0],
        Vector3F(1.0f, 0.0f, 0.0f),
        Vector3F(1.0f, 0.0f, 0.0f),
        material);

    auto collider = Collider<float>::builder()
                        .with_units(HostBuffer<Unit<float>> { make_plane_unit() })
                        .with_fluid(fluid)
                        .with_surface_interactions(
                            HostBuffer<MaxwellianSurfaceInteraction<float>> { interaction })
                        .build();

    collider.collide(1.0f);

    FluidInternalEnergy<float> actual {};
    atlas::copy_device_to_host(atlas::raw_pointer_cast(internal_energies.data().data()), &actual, 1);
    EXPECT_NEAR(actual.translational, expected.translational, tol);
    EXPECT_NEAR(actual.rotational, expected.rotational, tol);
    EXPECT_NEAR(actual.vibrational, expected.vibrational, tol);
}

TEST(Collider, CollidePreservesInternalEnergyForIsothermalSurfaceInteraction) {
    const auto fluid = make_fluid();
    fluid->set_particle_count(1);

    auto* positions = fluid->state<FluidPositionState<float>>();
    auto* velocities = fluid->state<FluidVelocityState<float>>();
    auto* species = fluid->state<FluidSpeciesState<float>>();
    auto& internal_energies = fluid->emplace_state<FluidInternalEnergyState<float>>(fluid->buffer_size());
    ASSERT_NE(positions, nullptr);
    ASSERT_NE(velocities, nullptr);
    ASSERT_NE(species, nullptr);

    const FluidInternalEnergy<float> incident_energy { 1.0f, 2.0f, 3.0f };
    positions->data()[0] = Vector3F(-1.0f, 0.0f, 0.0f);
    velocities->data()[0] = Vector3F(1.0f, 0.0f, 0.0f);
    species->data()[0] = 0u;
    internal_energies.data()[0] = incident_energy;
    fluid->particle_properties().push_back(
        MaterialProperties<float>::builder()
            .with_mass(1.0f)
            .with_molecular_mass(1.0f)
            .build());

    auto collider = Collider<float>::builder()
                        .with_units(HostBuffer<Unit<float>> { make_plane_unit() })
                        .with_fluid(fluid)
                        .with_surface_interactions(
                            HostBuffer<IsothermalSurfaceInteraction<float>> { make_specular_interaction() })
                        .build();

    collider.collide(1.0f);

    FluidInternalEnergy<float> actual {};
    atlas::copy_device_to_host(atlas::raw_pointer_cast(internal_energies.data().data()), &actual, 1);
    EXPECT_NEAR(actual.translational, incident_energy.translational, tol);
    EXPECT_NEAR(actual.rotational, incident_energy.rotational, tol);
    EXPECT_NEAR(actual.vibrational, incident_energy.vibrational, tol);
}
