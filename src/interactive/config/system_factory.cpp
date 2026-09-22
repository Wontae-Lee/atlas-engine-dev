#include "config/system_factory.h"

#include <atlas/atlas.h>

#include <stdexcept>
#include <utility>

namespace atlas::interactive {

namespace {

Float3
vec3(const SimulationConfig::Vec3& value) {
    return Float3(value[0], value[1], value[2]);
}

Quaternion
quat(const SimulationConfig::Quat& value) {
    return Quaternion(value[0], value[1], value[2], value[3]);
}

template <typename State, typename Value>
void
set_fluid_state(Fluid& fluid,
                const std::vector<Value>& values,
                const std::size_t particle_count) {
    if (values.empty()) return;
    if (values.size() != particle_count) {
        throw std::invalid_argument("Fluid initial state length must equal particle_count.");
    }
    State* state = fluid.state<State>();
    if (state == nullptr) state = &fluid.emplace_state<State>(fluid.buffer_size());
    atlas::copy_host_to_device(values.data(), state->data(), values.size());
}

template <typename State, typename Value>
void
set_optional_fluid_state(Fluid& fluid,
                         const std::vector<Value>& values,
                         const std::size_t particle_count) {
    if (values.size() != particle_count) {
        throw std::invalid_argument("Fluid initial state length must equal particle_count.");
    }
    State* state = fluid.state<State>();
    if (state == nullptr) state = &fluid.emplace_state<State>(fluid.buffer_size());
    if (!values.empty()) {
        atlas::copy_host_to_device(values.data(), state->data(), values.size());
    }
}

template <typename State, typename Value>
void
set_universe_state(Universe& universe, const std::optional<std::vector<Value>>& values) {
    if (!values) return;
    if (values->size() != static_cast<std::size_t>(universe.cell_count())) {
        throw std::invalid_argument("Universe initial state length must equal cell_count.");
    }
    State& state = universe.emplace_state<State>(values->size());
    atlas::copy_host_to_device(values->data(), state.data(), values->size());
}

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

Unit
make_unit(const SimulationConfig::Unit& config,
          std::vector<atlas::host_shared_ptr<TriangleMesh>>& mesh_owners,
          std::size_t& mesh_index) {
    const SyncHostPtr sync = Sync::builder()
                                 .with_rigid_pose(vec3(config.translation),
                                                  quat(config.orientation))
                                 .make_host_shared();
    auto builder = Unit::builder();
    builder.with_geometry(make_geometry(config.geometry, mesh_owners, mesh_index)).with_sync(sync);
    if (config.velocity) builder.with_velocity(vec3(*config.velocity));
    if (config.acceleration) builder.with_acceleration(vec3(*config.acceleration));
    if (config.angular_velocity) builder.with_angular_velocity(vec3(*config.angular_velocity));
    if (config.angular_acceleration) {
        builder.with_angular_acceleration(vec3(*config.angular_acceleration));
    }
    return builder.build();
}

SourceHostPtr
make_source(const SimulationConfig::Source& config,
            std::vector<atlas::host_shared_ptr<TriangleMesh>>& mesh_owners,
            std::size_t& mesh_index) {
    Unit unit = make_unit(config.unit, mesh_owners, mesh_index);
    if (config.kind == SimulationConfig::SourceKind::surface) {
        return atlas::make_host_shared<Source>(Source(SurfaceSource::builder()
                                                         .with_unit(std::move(unit))
                                                         .with_tolerance(config.tolerance)
                                                         .with_spacing(config.spacing)
                                                         .build()));
    }
    return atlas::make_host_shared<Source>(Source(VolumeSource::builder()
                                                     .with_unit(std::move(unit))
                                                     .with_tolerance(config.tolerance)
                                                     .with_spacing(config.spacing)
                                                     .build()));
}

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
        } else {
            throw std::invalid_argument(
                "Maxwell-Boltzmann generator requires materials or species_mass.");
        }
        return atlas::make_host_shared<Generator>(Generator(builder.build()));
    }
    }
    throw std::invalid_argument("Unknown generator type.");
}

}

SystemFactory::SystemFactory(SimulationConfig config)
    : _config(std::move(config)) {}

atlas::SystemHostPtr
SystemFactory::create() {
    std::size_t mesh_index = 0;

    MaterialDictionaryHostPtr materials;
    if (!_config.fluid.materials.empty()) {
        auto builder = MaterialDictionary::builder();
        for (const auto& material : _config.fluid.materials) {
            builder.with_material(make_material(material));
        }
        materials = builder.make_host_shared();
    }
    if (materials) {
        for (const std::size_t species : _config.fluid.species) {
            if (species >= _config.fluid.materials.size()) {
                throw std::invalid_argument("Fluid species id is outside the material dictionary.");
            }
        }
    }

    auto fluid = Fluid::builder()
                     .with_buffer_size(_config.fluid.buffer_size)
                     .with_particle_count(_config.fluid.particle_count)
                     .with_statistical_weight(_config.fluid.statistical_weight)
                     .with_materials(materials)
                     .make_host_unique();

    std::vector<Float3> positions;
    positions.reserve(_config.fluid.position.size());
    for (const auto& value : _config.fluid.position) positions.push_back(vec3(value));
    std::vector<Float3> velocities;
    velocities.reserve(_config.fluid.velocity.size());
    for (const auto& value : _config.fluid.velocity) velocities.push_back(vec3(value));
    set_fluid_state<FluidPositionState>(*fluid, positions, _config.fluid.particle_count);
    set_fluid_state<FluidVelocityState>(*fluid, velocities, _config.fluid.particle_count);
    set_fluid_state<FluidSpeciesState>(*fluid,
                                       _config.fluid.species,
                                       _config.fluid.particle_count);
    if (_config.fluid.temperature) {
        set_optional_fluid_state<FluidTemperatureState>(*fluid,
                                                        *_config.fluid.temperature,
                                                        _config.fluid.particle_count);
    }
    if (_config.fluid.translational_energy) {
        set_optional_fluid_state<FluidTranslationalEnergyState>(
            *fluid, *_config.fluid.translational_energy, _config.fluid.particle_count);
    }
    if (_config.fluid.rotational_energy) {
        set_optional_fluid_state<FluidRotationalEnergyState>(
            *fluid, *_config.fluid.rotational_energy, _config.fluid.particle_count);
    }
    if (_config.fluid.vibrational_energy) {
        set_optional_fluid_state<FluidVibrationalEnergyState>(
            *fluid, *_config.fluid.vibrational_energy, _config.fluid.particle_count);
    }

    auto universe_builder = Universe::builder();
    universe_builder.with_cell_size(_config.universe.cell_size);
    if (_config.universe.geometry) {
        universe_builder.with_geometry(
            make_geometry(*_config.universe.geometry, _mesh_owners, mesh_index));
    } else {
        universe_builder.with_lower_corner(vec3(_config.universe.lower_corner))
            .with_upper_corner(vec3(_config.universe.upper_corner));
    }
    auto universe = universe_builder.make_host_unique();

    set_universe_state<UniverseTemperatureState>(*universe, _config.universe.temperature);
    auto convert_vectors = [](const std::optional<std::vector<SimulationConfig::Vec3>>& values) {
        std::optional<std::vector<Float3>> converted;
        if (values) {
            converted.emplace();
            converted->reserve(values->size());
            for (const auto& value : *values) converted->push_back(vec3(value));
        }
        return converted;
    };
    set_universe_state<UniverseBulkVelocityState>(*universe,
                                                   convert_vectors(_config.universe.bulk_velocity));
    set_universe_state<UniverseFieldForceState>(*universe,
                                                 convert_vectors(_config.universe.field_force));
    set_universe_state<UniverseGravityState>(*universe,
                                              convert_vectors(_config.universe.gravity));
    set_universe_state<UniverseThermalEnergyState>(*universe,
                                                    _config.universe.thermal_energy);
    set_universe_state<UniverseKnudsenNumberState>(*universe,
                                                    _config.universe.knudsen_number);

    auto builder = System::builder();
    builder.with_fluid(std::move(fluid))
        .with_universe(std::move(universe))
        .with_dt(_config.dt);

    for (const auto& solver : _config.solvers) {
        DsmcKernelType kernel = DsmcKernelType::hard_sphere;
        if (solver.kernel == SimulationConfig::DsmcKernelKind::variable_hard_sphere) {
            kernel = DsmcKernelType::variable_hard_sphere;
        } else if (solver.kernel == SimulationConfig::DsmcKernelKind::variable_soft_sphere) {
            kernel = DsmcKernelType::variable_soft_sphere;
        }
        builder.with_solver(DsmcSolver::builder()
                                .with_kernel_type(kernel)
                                .with_majorant_sample_pairs(solver.majorant_sample_pairs)
                                .with_majorant_exhaustive_limit(solver.majorant_exhaustive_limit)
                                .make_host_shared());
    }
    for (const auto& emitter : _config.emitters) {
        builder.with_emitter(make_source(emitter.source, _mesh_owners, mesh_index),
                             make_generator(emitter.generator, materials));
    }
    for (const auto& collider : _config.colliders) {
        const DiffuseSampling sampling =
            collider.diffuse_sampling == SimulationConfig::DiffuseSamplingKind::cosine_weighted
                ? DiffuseSampling::cosine_weighted
                : DiffuseSampling::uniform;
        builder.with_collider(Collider(IsothermalCollider::builder()
                                           .with_unit(make_unit(collider.unit,
                                                                _mesh_owners,
                                                                mesh_index))
                                           .with_momentum_accommodation_coefficient(
                                               collider.momentum_accommodation_coefficient)
                                           .with_restitution(collider.restitution)
                                           .with_diffuse_sampling(sampling)
                                           .build()));
    }
    for (const auto& sink : _config.sinks) {
        Unit unit = make_unit(sink.unit, _mesh_owners, mesh_index);
        if (sink.kind == SimulationConfig::SinkKind::surface) {
            builder.with_sink(Sink(SurfaceSink::builder()
                                       .with_unit(std::move(unit))
                                       .with_tolerance(sink.tolerance)
                                       .build()));
        } else if (sink.kind == SimulationConfig::SinkKind::volume) {
            builder.with_sink(Sink(VolumeSink::builder()
                                       .with_unit(std::move(unit))
                                       .with_tolerance(sink.tolerance)
                                       .build()));
        } else {
            builder.with_sink(Sink(TracingSink::builder().with_unit(std::move(unit)).build()));
        }
    }
    if (_config.codec) {
        const auto& codec = *_config.codec;
        builder.with_codec(make_host_shared<Codec>(Codec(
            KnudsenCodec::builder()
                .with_representative_characteristic_length(
                    codec.representative_characteristic_length)
                .with_representative_collision_cross_sectional_area(
                    codec.representative_collision_cross_sectional_area)
                .with_representative_statistical_weight(codec.representative_statistical_weight)
                .with_representative_cell_volume(codec.representative_cell_volume)
                .build())));
    }
    return builder.make_host_unique();
}

const SimulationConfig&
SystemFactory::config() const noexcept {
    return _config;
}

}
