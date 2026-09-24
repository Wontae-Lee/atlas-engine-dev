/**
 * @file
 * @brief Implements Atlas core object construction from interactive configuration.
 */

#include "config/system_factory.h"

#include <atlas/atlas.h>

#include <stdexcept>
#include <utility>

namespace atlas::interactive {

namespace {

/// Converts a configuration vector to the Atlas math representation.
Float3
vec3(const SimulationConfig::Vec3& value) {
    return Float3(value[0], value[1], value[2]);
}

/// Converts a scalar-first configuration quaternion to Atlas.
Quaternion
quat(const SimulationConfig::Quat& value) {
    return Quaternion(value[0], value[1], value[2], value[3]);
}

/// Converts one material configuration into its Atlas tagged union.
Material
make_material(const SimulationConfig::Material& config) {
    const auto arguments = [&config](auto constructor) {
        return Material(constructor(config.mass,
                                    config.translational_energy,
                                    config.rotational_energy,
                                    config.vibrational_energy,
                                    config.reference_diameter,
                                    config.reference_temperature,
                                    config.viscosity_index,
                                    config.scattering_parameter));
    };
    switch (config.kind) {
    case SimulationConfig::MaterialKind::molecule:
        return arguments([](auto... values) { return Molecule(values...); });
    case SimulationConfig::MaterialKind::atom:
        return arguments([](auto... values) { return Atom(values...); });
    case SimulationConfig::MaterialKind::ion:
        return arguments([](auto... values) { return Ion(values...); });
    case SimulationConfig::MaterialKind::neutron:
        return arguments([](auto... values) { return Neutron(values...); });
    case SimulationConfig::MaterialKind::solid:
        return Material(Solid(config.mass));
    }
    throw std::invalid_argument("Unknown material type.");
}

/// Converts configured geometry while preserving triangle-mesh ownership.
Geometry
make_geometry(const SimulationConfig::Geometry& config,
              std::vector<atlas::host_shared_ptr<TriangleMesh>>& mesh_owners,
              std::size_t& mesh_index) {
    switch (config.kind) {
    case SimulationConfig::GeometryKind::box:
        return Geometry(Box::builder()
                            .with_lower_corner(vec3(config.lower_corner))
                            .with_upper_corner(vec3(config.upper_corner))
                            .build());
    case SimulationConfig::GeometryKind::circle:
        return Geometry(Circle::builder()
                            .with_center(vec3(config.center))
                            .with_normal(vec3(config.normal))
                            .with_radius(config.radius)
                            .build());
    case SimulationConfig::GeometryKind::cylinder:
        return Geometry(Cylinder::builder()
                            .with_center(vec3(config.center))
                            .with_radius(config.radius)
                            .with_height(config.height)
                            .with_open(config.open)
                            .build());
    case SimulationConfig::GeometryKind::plane:
        return Geometry(Plane::builder()
                            .with_normal_offset(vec3(config.normal), config.offset)
                            .build());
    case SimulationConfig::GeometryKind::sphere:
        return Geometry(Sphere::builder()
                            .with_center(vec3(config.center))
                            .with_radius(config.radius)
                            .build());
    case SimulationConfig::GeometryKind::square:
        return Geometry(Square::builder()
                            .with_center(vec3(config.center))
                            .with_normal(vec3(config.normal))
                            .with_side_length(config.side_length)
                            .build());
    case SimulationConfig::GeometryKind::triangle:
        return Geometry(Triangle::builder()
                            .with_vertices(vec3(config.a), vec3(config.b), vec3(config.c))
                            .build());
    case SimulationConfig::GeometryKind::triangle_mesh: {
        // Geometry stores a device view, so the factory keeps each owning mesh alive.
        if (mesh_index == mesh_owners.size()) {
            HostBuffer<TriangleContainer4> triangles;
            triangles.reserve(config.triangles.size());
            for (const auto& vertices : config.triangles) {
                const Triangle triangle = Triangle::builder()
                                              .with_vertices(vec3(vertices[0]),
                                                             vec3(vertices[1]),
                                                             vec3(vertices[2]))
                                              .build();
                triangles.push_back(
                    TriangleContainer4(triangle.a, triangle.b, triangle.c, triangle.normal));
            }
            mesh_owners.push_back(TriangleMesh::builder()
                                      .with_triangles(std::move(triangles))
                                      .make_host_shared());
        }
        return mesh_owners.at(mesh_index++)->make_device_geometry_view();
    }
    case SimulationConfig::GeometryKind::polygonal_prism:
        return Geometry(PolygonalPrism::builder()
                            .with_center(vec3(config.center))
                            .with_side_count(config.side_count)
                            .with_radius(config.radius)
                            .with_height(config.height)
                            .build());
    }
    throw std::invalid_argument("Unknown geometry type.");
}

/// Builds a particle generator and connects material-dependent sampling.
GeneratorHostPtr
make_generator(const SimulationConfig::Generator& config,
               const MaterialDictionaryHostPtr& materials) {
    const HostBuffer<float> species_ratios(config.species_ratios.begin(),
                                           config.species_ratios.end());
    const HostBuffer<float> species_numbers(config.species_numbers.begin(),
                                            config.species_numbers.end());
    switch (config.kind) {
    case SimulationConfig::GeneratorKind::uniform:
        return atlas::make_host_shared<Generator>(Generator(UniformGenerator::builder()
            .with_species_ratios(species_ratios)
            .with_species_numbers(species_numbers)
            .with_temperature(config.temperature)
            .with_min_value(config.min_value)
            .with_max_value(config.max_value)
            .with_bulk_velocity(vec3(config.bulk_velocity))
            .with_seed(config.seed)
            .build()));
    case SimulationConfig::GeneratorKind::jittering:
        return atlas::make_host_shared<Generator>(Generator(JitteringGenerator::builder()
            .with_species_ratios(species_ratios)
            .with_species_numbers(species_numbers)
            .with_temperature(config.temperature)
            .with_base_value(config.base_value)
            .with_jitter_radius(config.jitter_radius)
            .with_bulk_velocity(vec3(config.bulk_velocity))
            .with_seed(config.seed)
            .build()));
    case SimulationConfig::GeneratorKind::maxwell_sigma:
        return atlas::make_host_shared<Generator>(Generator(MaxwellSigmaGenerator::builder()
            .with_species_ratios(species_ratios)
            .with_species_numbers(species_numbers)
            .with_temperature(config.temperature)
            .with_sigma(config.sigma)
            .with_bulk_velocity(vec3(config.bulk_velocity))
            .with_seed(config.seed)
            .build()));
    case SimulationConfig::GeneratorKind::maxwell_boltzmann: {
        auto builder = MaxwellBoltzmannGenerator::builder();
        builder.with_species_ratios(species_ratios)
            .with_species_numbers(species_numbers)
            .with_temperature(config.temperature)
            .with_bulk_velocity(vec3(config.bulk_velocity))
            .with_seed(config.seed);
        if (!config.species_mass.empty()) {
            builder.with_species_mass(
                HostBuffer<float>(config.species_mass.begin(), config.species_mass.end()));
        } else if (materials) {
            builder.with_material_dictionary(*materials);
        }
        return atlas::make_host_shared<Generator>(Generator(builder.build()));
    }
    }
    throw std::invalid_argument("Unknown generator type.");
}

}

CoreFactory::CoreFactory(SimulationConfig config)
    : _config(std::move(config)) {}

Material
CoreFactory::build(const SimulationConfig::Material& config) {
    return make_material(config);
}

Geometry
CoreFactory::build(const SimulationConfig::Geometry& config) {
    return make_geometry(config, _mesh_owners, _mesh_index);
}

Unit
CoreFactory::build(const SimulationConfig::Unit& config) {
    const SyncHostPtr sync = Sync::builder()
                                 .with_rigid_pose(vec3(config.translation), quat(config.orientation))
                                 .make_host_shared();
    auto builder = Unit::builder();
    builder.with_geometry(build(config.geometry)).with_sync(sync);
    if (config.velocity) builder.with_velocity(vec3(*config.velocity));
    if (config.acceleration) builder.with_acceleration(vec3(*config.acceleration));
    if (config.angular_velocity) builder.with_angular_velocity(vec3(*config.angular_velocity));
    if (config.angular_acceleration) {
        builder.with_angular_acceleration(vec3(*config.angular_acceleration));
    }
    return builder.build();
}

MaterialDictionaryHostPtr
CoreFactory::build_materials(const std::vector<SimulationConfig::Material>& configs) {
    if (configs.empty()) return {};
    auto builder = MaterialDictionary::builder();
    for (const auto& config : configs) builder.with_material(build(config));
    return builder.make_host_shared();
}

FluidHostPtr
CoreFactory::build(const SimulationConfig::Fluid& config) {
    return build_fluid(config, build_materials(config.materials));
}

FluidHostPtr
CoreFactory::build_fluid(const SimulationConfig::Fluid& config,
                         const MaterialDictionaryHostPtr& materials) {
    auto builder = Fluid::builder()
                       .with_buffer_size(config.buffer_size)
                       .with_particle_count(config.particle_count)
                       .with_statistical_weight(config.statistical_weight)
                       .with_materials(materials);
    if (config.position_provided || !config.position.empty()) {
        HostBuffer<Float3> values;
        for (const auto& value : config.position) values.push_back(vec3(value));
        builder.with_position(std::move(values));
    }
    if (config.velocity_provided || !config.velocity.empty()) {
        HostBuffer<Float3> values;
        for (const auto& value : config.velocity) values.push_back(vec3(value));
        builder.with_velocity(std::move(values));
    }
    if (config.species_provided || !config.species.empty()) {
        builder.with_species(HostBuffer<std::size_t>(config.species.begin(),
                                                      config.species.end()));
    }
    if (config.temperature) {
        builder.with_temperature(HostBuffer<float>(config.temperature->begin(),
                                                    config.temperature->end()));
    }
    if (config.translational_energy) {
        builder.with_translational_energy(HostBuffer<float>(
            config.translational_energy->begin(), config.translational_energy->end()));
    }
    if (config.rotational_energy) {
        builder.with_rotational_energy(HostBuffer<float>(config.rotational_energy->begin(),
                                                          config.rotational_energy->end()));
    }
    if (config.vibrational_energy) {
        builder.with_vibrational_energy(HostBuffer<float>(config.vibrational_energy->begin(),
                                                           config.vibrational_energy->end()));
    }
    return builder.make_host_unique();
}

UniverseHostPtr
CoreFactory::build(const SimulationConfig::Universe& config) {
    auto builder = Universe::builder();
    builder.with_cell_size(config.cell_size);
    if (config.geometry) {
        builder.with_geometry(build(*config.geometry));
    } else {
        builder.with_lower_corner(vec3(config.lower_corner))
            .with_upper_corner(vec3(config.upper_corner));
    }
    if (config.temperature) {
        builder.with_temperature(HostBuffer<float>(config.temperature->begin(),
                                                    config.temperature->end()));
    }
    const auto vectors = [](const std::vector<SimulationConfig::Vec3>& source) {
        HostBuffer<Float3> values;
        for (const auto& value : source) values.push_back(vec3(value));
        return values;
    };
    if (config.bulk_velocity) builder.with_bulk_velocity(vectors(*config.bulk_velocity));
    if (config.field_force) builder.with_field_force(vectors(*config.field_force));
    if (config.gravity) builder.with_gravity(vectors(*config.gravity));
    if (config.thermal_energy) {
        builder.with_thermal_energy(HostBuffer<float>(config.thermal_energy->begin(),
                                                       config.thermal_energy->end()));
    }
    if (config.knudsen_number) {
        builder.with_knudsen_number(HostBuffer<float>(config.knudsen_number->begin(),
                                                       config.knudsen_number->end()));
    }
    return builder.make_host_unique();
}

SolverHostPtr
CoreFactory::build(const SimulationConfig::Solver& config) {
    DsmcKernelType kernel = DsmcKernelType::hard_sphere;
    if (config.kernel == SimulationConfig::DsmcKernelKind::variable_hard_sphere) {
        kernel = DsmcKernelType::variable_hard_sphere;
    } else if (config.kernel == SimulationConfig::DsmcKernelKind::variable_soft_sphere) {
        kernel = DsmcKernelType::variable_soft_sphere;
    }
    return DsmcSolver::builder()
        .with_kernel_type(kernel)
        .with_majorant_sample_pairs(config.majorant_sample_pairs)
        .with_majorant_exhaustive_limit(config.majorant_exhaustive_limit)
        .make_host_shared();
}

SourceHostPtr
CoreFactory::build(const SimulationConfig::Source& config) {
    Unit unit = build(config.unit);
    if (config.kind == SimulationConfig::SourceKind::surface) {
        return make_host_shared<Source>(Source(SurfaceSource::builder()
            .with_unit(std::move(unit)).with_tolerance(config.tolerance)
            .with_spacing(config.spacing).build()));
    }
    return make_host_shared<Source>(Source(VolumeSource::builder()
        .with_unit(std::move(unit)).with_tolerance(config.tolerance)
        .with_spacing(config.spacing).build()));
}

GeneratorHostPtr
CoreFactory::build(const SimulationConfig::Generator& config,
                   const MaterialDictionaryHostPtr& materials) {
    return make_generator(config, materials);
}

Collider
CoreFactory::build(const SimulationConfig::Collider& config) {
    const DiffuseSampling sampling =
        config.diffuse_sampling == SimulationConfig::DiffuseSamplingKind::cosine_weighted
            ? DiffuseSampling::cosine_weighted
            : DiffuseSampling::uniform;
    return Collider(IsothermalCollider::builder()
        .with_unit(build(config.unit))
        .with_momentum_accommodation_coefficient(config.momentum_accommodation_coefficient)
        .with_restitution(config.restitution)
        .with_diffuse_sampling(sampling).build());
}

Sink
CoreFactory::build(const SimulationConfig::Sink& config) {
    Unit unit = build(config.unit);
    if (config.kind == SimulationConfig::SinkKind::surface) {
        return Sink(SurfaceSink::builder().with_unit(std::move(unit))
            .with_tolerance(config.tolerance).build());
    }
    if (config.kind == SimulationConfig::SinkKind::volume) {
        return Sink(VolumeSink::builder().with_unit(std::move(unit))
            .with_tolerance(config.tolerance).build());
    }
    return Sink(TracingSink::builder().with_unit(std::move(unit)).build());
}

CodecHostPtr
CoreFactory::build(const SimulationConfig::Codec& config) {
    return make_host_shared<Codec>(Codec(KnudsenCodec::builder()
        .with_representative_characteristic_length(config.representative_characteristic_length)
        .with_representative_collision_cross_sectional_area(config.representative_collision_cross_sectional_area)
        .with_representative_statistical_weight(config.representative_statistical_weight)
        .with_representative_cell_volume(config.representative_cell_volume).build()));
}

atlas::SystemHostPtr
CoreFactory::build(const SimulationConfig& config) {
    _mesh_index = 0;
    const MaterialDictionaryHostPtr materials = build_materials(config.fluid.materials);
    auto fluid = build_fluid(config.fluid, materials);
    auto universe = build(config.universe);
    auto builder = System::builder();
    builder.with_fluid(std::move(fluid)).with_universe(std::move(universe)).with_dt(config.dt);
    for (const auto& solver : config.solvers) builder.with_solver(build(solver));
    for (const auto& emitter : config.emitters) {
        builder.with_emitter(build(emitter.source), build(emitter.generator, materials));
    }
    for (const auto& collider : config.colliders) builder.with_collider(build(collider));
    for (const auto& sink : config.sinks) builder.with_sink(build(sink));
    if (config.codec) builder.with_codec(build(*config.codec));
    return builder.make_host_unique();
}

atlas::SystemHostPtr
CoreFactory::create() {
    return build(_config);
}

const SimulationConfig&
CoreFactory::config() const noexcept {
    return _config;
}

}
