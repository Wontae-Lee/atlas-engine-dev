/**
 * @file
 * @brief Provides a reference construction surface for native Atlas simulations.
 */

#pragma once

#include <atlas/atlas.h>

#include <cstddef>
#include <utility>

namespace atlas_examples::template_case {

/**
 * @brief Constructs a System demonstrating the public native API surface.
 *
 * The returned system intentionally combines several alternative components so
 * new examples can copy the relevant pieces and remove the rest.
 */
inline atlas::System
build_system() {
    constexpr std::size_t capacity = 1024;
    constexpr std::size_t particle_count = 2;

    auto materials = atlas::MaterialDictionary::builder()
        .with_material(atlas::Material(atlas::Molecule(
            4.65e-26f, 0.0f, 0.0f, 0.0f, 4.17e-10f, 273.0f, 0.74f, 1.0f)))
        // Atom, Ion, and Neutron accept the same eight physical values.
        .with_material(atlas::Material(atlas::Atom(
            4.65e-26f, 0.0f, 0.0f, 0.0f, 4.17e-10f, 273.0f, 0.74f, 1.0f)))
        .with_material(atlas::Material(atlas::Ion(
            4.65e-26f, 0.0f, 0.0f, 0.0f, 4.17e-10f, 273.0f, 0.74f, 1.0f)))
        .with_material(atlas::Material(atlas::Neutron(
            4.65e-26f, 0.0f, 0.0f, 0.0f, 4.17e-10f, 273.0f, 0.74f, 1.0f)))
        .with_material(atlas::Material(atlas::Solid(1.0f)))
        .make_host_shared();

    auto fluid = atlas::Fluid::builder()
        .with_buffer_size(capacity)
        .with_particle_count(particle_count)
        .with_statistical_weight(1.0f)
        .with_materials(materials)
        .make_host_unique();
    fluid->state<atlas::FluidPositionState>()->data() = atlas::DeviceBuffer<atlas::Float3>(
        capacity, atlas::Float3(-0.5f, 0.0f, 0.0f));
    fluid->state<atlas::FluidVelocityState>()->data() = atlas::DeviceBuffer<atlas::Float3>(
        capacity, atlas::Float3(10.0f, 0.0f, 0.0f));
    fluid->state<atlas::FluidSpeciesState>()->data() =
        atlas::DeviceBuffer<std::size_t>(capacity, 0);
    fluid->emplace_state<atlas::FluidTemperatureState>(capacity);
    fluid->emplace_state<atlas::FluidTranslationalEnergyState>(capacity);
    fluid->emplace_state<atlas::FluidRotationalEnergyState>(capacity);
    fluid->emplace_state<atlas::FluidVibrationalEnergyState>(capacity);

    auto universe = atlas::Universe::builder()
        .with_lower_corner(atlas::Float3(-1.0f))
        .with_upper_corner(atlas::Float3(1.0f))
        // Replace the corners with .with_geometry(geometry) to derive the domain from a bound.
        .with_cell_size(0.25f)
        .make_host_unique();
    universe->emplace_state<atlas::UniverseTemperatureState>(universe->cell_count());
    universe->emplace_state<atlas::UniverseBulkVelocityState>(universe->cell_count());
    universe->emplace_state<atlas::UniverseFieldForceState>(universe->cell_count());
    universe->emplace_state<atlas::UniverseGravityState>(universe->cell_count());
    universe->emplace_state<atlas::UniverseThermalEnergyState>(universe->cell_count());
    universe->emplace_state<atlas::UniverseKnudsenNumberState>(universe->cell_count());

    const atlas::Geometry source_geometry(atlas::Box::builder()
        .with_lower_corner(atlas::Float3(-0.9f, -0.2f, -0.2f))
        .with_upper_corner(atlas::Float3(-0.8f, 0.2f, 0.2f))
        .build());
    auto source_sync = atlas::Sync::builder()
        .with_rigid_pose(atlas::Float3(0.0f), atlas::Quaternion())
        .make_host_shared();
    const atlas::Unit source_unit = atlas::Unit::builder()
        .with_geometry(source_geometry)
        .with_sync(source_sync)
        .with_velocity(atlas::Float3(0.0f))
        .with_acceleration(atlas::Float3(0.0f))
        .with_angular_velocity(atlas::Float3(0.0f))
        .with_angular_acceleration(atlas::Float3(0.0f))
        .build();
    auto source = atlas::make_host_shared<atlas::Source>(atlas::Source(
        atlas::VolumeSource::builder()
            .with_unit(source_unit)
            .with_spacing(0.1f)
            .with_tolerance(0.0f)
            .build()));
    // SurfaceSource uses the same Unit, spacing, and tolerance builder shape.

    auto generator = atlas::make_host_shared<atlas::Generator>(atlas::Generator(
        atlas::MaxwellBoltzmannGenerator::builder()
            .with_species_ratios({ 1.0f })
            .with_species_numbers({ 0.0f })
            .with_material_dictionary(*materials)
            .with_temperature(300.0f)
            .with_bulk_velocity(atlas::Float3(10.0f, 0.0f, 0.0f))
            .with_seed(1)
            .build()));
    // UniformGenerator adds min/max; JitteringGenerator adds base/jitter radius;
    // MaxwellSigmaGenerator adds sigma. All share ratios, numbers, temperature,
    // bulk velocity, and seed. MaxwellBoltzmann may use explicit species masses.

    const atlas::Geometry cylinder(atlas::Cylinder::builder()
        .with_center(atlas::Float3(0.0f))
        .with_radius(0.2f)
        .with_height(1.0f)
        .with_open(true)
        .build());
    const atlas::Unit moving_unit = atlas::Unit::builder()
        .with_geometry(cylinder)
        .with_sync(atlas::Sync::builder().make_host_shared())
        .with_velocity(atlas::Float3(0.1f, 0.0f, 0.0f))
        .with_angular_velocity(atlas::Float3(0.0f, 0.0f, 0.1f))
        .build();
    const atlas::Collider collider(atlas::IsothermalCollider::builder()
        .with_unit(moving_unit)
        .with_momentum_accommodation_coefficient(1.0f)
        .with_restitution(1.0f)
        .with_diffuse_sampling(atlas::DiffuseSampling::uniform)
        .build());

    const atlas::Sink surface_sink(atlas::SurfaceSink::builder()
        .with_unit(moving_unit).with_tolerance(0.0f).build());
    const atlas::Sink volume_sink(atlas::VolumeSink::builder()
        .with_unit(source_unit).with_tolerance(0.0f).build());
    const atlas::Sink tracing_sink(atlas::TracingSink::builder()
        .with_unit(moving_unit).build());

    auto solver = atlas::DsmcSolver::builder()
        .with_kernel_type(atlas::DsmcKernelType::variable_hard_sphere)
        .with_majorant_sample_pairs(8)
        .with_majorant_exhaustive_limit(5)
        .make_host_shared();
    auto codec = atlas::make_host_shared<atlas::Codec>(atlas::Codec(
        atlas::KnudsenCodec::builder()
            .with_representative_characteristic_length(1.0f)
            .with_representative_collision_cross_sectional_area(1.0f)
            .with_representative_statistical_weight(1.0f)
            .with_representative_cell_volume(1.0f)
            .build()));

    return atlas::System::builder()
        .with_fluid(std::move(fluid))
        .with_universe(std::move(universe))
        .with_dt(1.0e-4f)
        .with_solver(std::move(solver))
        .with_emitter(std::move(source), std::move(generator))
        .with_collider(collider)
        .with_sink(surface_sink)
        .with_sink(volume_sink)
        .with_sink(tracing_sink)
        .with_codec(std::move(codec))
        .build();
}

// Geometry builder alternatives:
//
// atlas::Box::builder().with_lower_corner(...).with_upper_corner(...)
// atlas::Circle::builder().with_center(...).with_normal(...).with_radius(...)
// atlas::Cylinder::builder().with_center(...).with_radius(...).with_height(...).with_open(...)
// atlas::Plane::builder().with_normal_offset(...)
// atlas::Sphere::builder().with_center(...).with_radius(...)
// atlas::Square::builder().with_center(...).with_normal(...).with_side_length(...)
// atlas::Triangle::builder().with_vertices(...)
// atlas::TriangleMesh::builder().load_from_obj("mesh.obj")
// atlas::PolygonalPrism::builder().with_center(...).with_side_count(...).with_radius(...).with_height(...)

}
