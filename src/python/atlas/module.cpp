#include <nanobind/nanobind.h>
#include <nanobind/operators.h>
#include <nanobind/stl/filesystem.h>
#include <nanobind/stl/optional.h>
#include <nanobind/stl/shared_ptr.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/vector.h>

#include <atlas/buffer/host_buffer.h>
#include <atlas/collider/collider.h>
#include <atlas/collider/interaction/isothermal_surface_kernel.h>
#include <atlas/collider/interaction/maxwellian_surface_interaction.h>
#include <atlas/fluid/fluid.h>
#include <atlas/generator/generator.h>
#include <atlas/generator/maxwell_boltzmann_generator.h>
#include <atlas/geometry/box.h>
#include <atlas/geometry/circle.h>
#include <atlas/geometry/cylinder.h>
#include <atlas/geometry/geometry.h>
#include <atlas/geometry/plane.h>
#include <atlas/geometry/sphere.h>
#include <atlas/geometry/square.h>
#include <atlas/geometry/triangle.h>
#include <atlas/geometry/triangle_mesh.h>
#include <atlas/material/material_properties.h>
#include <atlas/math/matrix/float3x3.h>
#include <atlas/math/quaternion.h>
#include <atlas/math/vector/float3.h>
#include <atlas/math/vector/int3.h>
#include <atlas/observer/observer.h>
#include <atlas/observer/sensor_metrics.h>
#include <atlas/orchestrator/orchestrator.h>
#include <atlas/searcher/spatial_hashing_searcher.h>
#include <atlas/sink/sink.h>
#include <atlas/solver/dsmc/dsmc_solver.h>
#include <atlas/solver/sph/sph_solver.h>
#include <atlas/source/source.h>
#include <atlas/sync/sync.h>
#include <atlas/system/system.h>
#include <atlas/unit/unit.h>
#include <atlas/universe/universe.h>

#include <cstddef>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace nb = nanobind;

using nb::literals::operator""_a;

namespace {

const char*
atlas_backend() {
#if defined(ATLAS_TASKING_CUDA)
    return "cuda";
#else
    return "tbb";
#endif
}

std::string
vector3_repr(const atlas::Float3& v) {
    return "Float3(" + std::to_string(v.x) + ", " + std::to_string(v.y) + ", "
           + std::to_string(v.z) + ")";
}

std::string
vector3i_repr(const atlas::Int3& v) {
    return "Int3(" + std::to_string(v.x) + ", " + std::to_string(v.y) + ", "
           + std::to_string(v.z) + ")";
}

std::string
quaternion_repr(const atlas::Quaternion& q) {
    return "Quaternion(" + std::to_string(q.w) + ", " + std::to_string(q.x) + ", "
           + std::to_string(q.y) + ", " + std::to_string(q.z) + ")";
}

atlas::GeneratorHostPtr
make_maxwell_boltzmann_generator(float temperature,
                                 float molecular_mass,
                                 const atlas::Float3& bulk_velocity,
                                 unsigned int seed) {
    return atlas::MaxwellBoltzmannGenerator::builder()
        .with_temperature(temperature)
        .with_molecular_mass(molecular_mass)
        .with_bulk_velocity(bulk_velocity)
        .with_seed(seed)
        .make_host_shared();
}

atlas::UniverseHostPtr
make_universe(const atlas::Float3& lower_corner,
              const atlas::Float3& upper_corner,
              float cell_size,
              const std::vector<atlas::Unit>& source_units,
              const std::vector<atlas::Unit>& sink_units,
              const std::vector<atlas::Unit>& collider_units) {
    // Units are owned centrally by the Universe (one UnitField per consumer
    // role); Source/Sink/Collider borrow the matching field via with_universe.
    return atlas::Universe::builder()
        .with_lower_corner(lower_corner)
        .with_upper_corner(upper_corner)
        .with_cell_size(cell_size)
        .with_source_units(atlas::HostBuffer<atlas::Unit>(source_units.begin(), source_units.end()))
        .with_sink_units(atlas::HostBuffer<atlas::Unit>(sink_units.begin(), sink_units.end()))
        .with_collider_units(atlas::HostBuffer<atlas::Unit>(collider_units.begin(), collider_units.end()))
        .make_host_shared();
}

atlas::FluidHostPtr
make_fluid(std::size_t buffer_size,
           const std::vector<atlas::MaterialProperties>& properties,
           const std::vector<atlas::GeneratorHostPtr>& generators) {
    return atlas::Fluid::builder()
        .with_buffer_size(buffer_size)
        .with_properties(atlas::HostBuffer<atlas::MaterialProperties>(properties.begin(), properties.end()))
        .with_generators(atlas::HostBuffer<atlas::GeneratorHostPtr>(generators.begin(), generators.end()))
        .make_host_shared();
}

atlas::SearcherHostPtr
make_spatial_hashing_searcher(const atlas::UniverseHostPtr& universe,
                              const atlas::FluidHostPtr& fluid) {
    return atlas::SpatialHashingSearcher::builder()
        .with_universe(universe)
        .with_fluid(fluid)
        .make_host_shared();
}

atlas::Geometry
make_box(const atlas::Float3& lower_corner, const atlas::Float3& upper_corner) {
    return atlas::Geometry(*atlas::Box::builder()
                                .with_lower_corner(lower_corner)
                                .with_upper_corner(upper_corner)
                                .make_host_shared());
}

atlas::Geometry
make_sphere(const atlas::Float3& center, float radius) {
    return atlas::Geometry(*atlas::Sphere::builder()
                                .with_center(center)
                                .with_radius(radius)
                                .make_host_shared());
}

atlas::Geometry
make_cylinder(const atlas::Float3& center, float radius, float height, bool open) {
    return atlas::Geometry(*atlas::Cylinder::builder()
                                .with_center(center)
                                .with_radius(radius)
                                .with_height(height)
                                .with_open(open)
                                .make_host_shared());
}

atlas::Geometry
make_circle(const atlas::Float3& center, const atlas::Float3& normal, float radius) {
    return atlas::Geometry(*atlas::Circle::builder()
                                .with_center(center)
                                .with_normal(normal)
                                .with_radius(radius)
                                .make_host_shared());
}

atlas::Geometry
make_plane(const atlas::Float3& point, const atlas::Float3& normal) {
    return atlas::Geometry(*atlas::Plane::builder()
                                .with_point_normal(point, normal)
                                .make_host_shared());
}

atlas::Geometry
make_square(const atlas::Float3& center, const atlas::Float3& normal, float side_length) {
    return atlas::Geometry(*atlas::Square::builder()
                                .with_center(center)
                                .with_normal(normal)
                                .with_side_length(side_length)
                                .make_host_shared());
}

atlas::Geometry
make_triangle(const atlas::Float3& a, const atlas::Float3& b, const atlas::Float3& c) {
    return atlas::Geometry(*atlas::Triangle::builder()
                                .with_vertices(a, b, c)
                                .make_host_shared());
}

atlas::Geometry
make_triangle_mesh_from_obj(const std::string& filename) {
    // A triangle mesh's Geometry is a pointer VIEW into the mesh's device
    // buffers, so the mesh must outlive every Unit/UnitField that holds the
    // view. Keep every mesh built through the bindings alive for the module's
    // lifetime so the view never dangles.
    static std::vector<atlas::host_shared_ptr<atlas::TriangleMesh>> mesh_registry;

    auto builder = atlas::TriangleMesh::builder();
    builder.load_from_obj(filename);
    auto mesh = builder.make_host_shared();
    mesh_registry.push_back(mesh);

    return mesh->make_device_geometry_view();
}

atlas::ObserverHostPtr
make_observer(std::size_t reserve_count) {
    return atlas::Observer::builder()
        .with_source_sensor_metrics(reserve_count)
        .with_sink_sensor_metrics(reserve_count)
        .make_host_shared();
}

std::vector<atlas::SensorMetrics::Record>
observer_source_records(const atlas::Observer& observer) {
    const auto* metrics = observer.sensor_metrics<atlas::SourceSensorMetrics>();
    if (metrics == nullptr) {
        return {};
    }
    const auto& records = metrics->records();
    return std::vector<atlas::SensorMetrics::Record>(records.begin(), records.end());
}

std::vector<atlas::SensorMetrics::Record>
observer_sink_records(const atlas::Observer& observer) {
    const auto* metrics = observer.sensor_metrics<atlas::SinkSensorMetrics>();
    if (metrics == nullptr) {
        return {};
    }
    const auto& records = metrics->records();
    return std::vector<atlas::SensorMetrics::Record>(records.begin(), records.end());
}

atlas::SyncHostPtr
make_sync(const atlas::Float3& translation, const atlas::Quaternion& orientation) {
    return atlas::Sync::builder()
        .with_rigid_pose(translation, orientation)
        .make_host_shared();
}

atlas::Unit
make_unit(const atlas::Geometry& geometry, const atlas::SyncHostPtr& sync) {
    return atlas::Unit::builder()
        .with_geometry(geometry)
        .with_sync(sync)
        .build();
}

atlas::SourceHostPtr
make_source(const atlas::FluidHostPtr& fluid,
            const atlas::UniverseHostPtr& universe,
            atlas::SpawnType spawn_type,
            float spacing,
            float temperature,
            const atlas::ObserverHostPtr& observer) {
    return atlas::Source::builder()
        .with_universe(universe)
        .with_fluid(fluid)
        .with_observer(observer)
        .with_spawn_types(atlas::HostBuffer<atlas::SpawnType>(1, spawn_type))
        .with_spawn_operator(atlas::Spawn(spawn_type))
        .with_spacing(spacing)
        .with_temperature(temperature)
        .make_host_shared();
}

atlas::SinkHostPtr
make_sink(const atlas::FluidHostPtr& fluid,
          const atlas::UniverseHostPtr& universe,
          atlas::DespawnType despawn_type,
          bool flip,
          const atlas::ObserverHostPtr& observer) {
    return atlas::Sink::builder()
        .with_universe(universe)
        .with_fluid(fluid)
        .with_observer(observer)
        .with_despawn_types(atlas::HostBuffer<atlas::DespawnType>(1, despawn_type))
        .with_despawn_operator(atlas::Despawn(despawn_type))
        .with_flip(flip)
        .make_host_shared();
}

atlas::IsothermalSurfaceInteraction
make_isothermal_interaction(atlas::DiffuseSampling diffuse_sampling,
                            float restitution,
                            float momentum_acc,
                            float temperature) {
    return atlas::IsothermalSurfaceInteraction::builder()
        .with_diffuse_sampling(diffuse_sampling)
        .with_restitution(restitution)
        .with_momentum_acc(momentum_acc)
        .with_temperature(temperature)
        .build();
}

atlas::ColliderHostPtr
make_collider(const atlas::FluidHostPtr& fluid,
              const atlas::UniverseHostPtr& universe,
              const atlas::IsothermalSurfaceInteraction& interaction,
              bool flip) {
    return atlas::Collider::builder()
        .with_fluid(fluid)
        .with_universe(universe)
        .with_surface_interactions(atlas::HostBuffer<atlas::IsothermalSurfaceInteraction>(1, interaction))
        .with_flip(flip)
        .make_host_shared();
}

atlas::MaxwellianSurfaceInteraction
make_maxwellian_interaction(float temperature,
                            float molecular_mass,
                            float momentum_acc,
                            float trans_acc,
                            float rot_acc,
                            float vib_acc) {
    return atlas::MaxwellianSurfaceInteraction::builder()
        .with_temperature(temperature)
        .with_molecular_mass(molecular_mass)
        .with_accommodation(momentum_acc, trans_acc, rot_acc, vib_acc)
        .build();
}

atlas::ColliderHostPtr
make_collider_maxwellian(const atlas::FluidHostPtr& fluid,
                         const atlas::UniverseHostPtr& universe,
                         const atlas::MaxwellianSurfaceInteraction& interaction,
                         bool flip) {
    return atlas::Collider::builder()
        .with_fluid(fluid)
        .with_universe(universe)
        .with_surface_interactions(atlas::HostBuffer<atlas::MaxwellianSurfaceInteraction>(1, interaction))
        .with_flip(flip)
        .make_host_shared();
}

struct SolverSpec {
    std::string name;
};

class SimpleSystem {
public:
    SimpleSystem(atlas::FluidHostPtr fluid, atlas::UniverseHostPtr universe)
        : _fluid(std::move(fluid))
        , _universe(std::move(universe)) {
        _searcher = make_spatial_hashing_searcher(_universe, _fluid);
    }

    void
    set_solver(const SolverSpec& spec) {
        if (spec.name == "dsmc") {
            _solver = atlas::make_host_shared<atlas::DsmcSolver>(
                _universe, _fluid, _searcher, atlas::DsmcKernelType::hard_sphere);
        } else if (spec.name == "sph") {
            _solver = atlas::make_host_shared<atlas::SphSolver>(
                _universe, _fluid, _searcher, atlas::SphKernelType::standard);
        } else {
            throw std::invalid_argument(
                "System: unknown solver '" + spec.name + "' (expected 'dsmc' or 'sph')");
        }

        _orchestrator = atlas::Orchestrator::builder()
                            .with_universe(_universe)
                            .with_fluid(_fluid)
                            .with_searcher(_searcher)
                            .with_solver(_solver)
                            .make_host_shared();
        _spec = spec;
    }

    const SolverSpec&
    solver() const {
        return _spec;
    }

    void
    set_source(atlas::SourceHostPtr source) {
        _source = std::move(source);
    }

    atlas::SourceHostPtr
    source() const {
        return _source;
    }

    void
    set_sink(atlas::SinkHostPtr sink) {
        _sink = std::move(sink);
    }

    atlas::SinkHostPtr
    sink() const {
        return _sink;
    }

    void
    set_collider(atlas::ColliderHostPtr collider) {
        _collider = std::move(collider);
    }

    atlas::ColliderHostPtr
    collider() const {
        return _collider;
    }

    void
    update(float dt) {
        const auto system = atlas::System::builder()
                                .with_fluid(_fluid)
                                .with_domain(_universe)
                                .with_source(_source)
                                .with_sink(_sink)
                                .with_collider(_collider)
                                .with_solver(_orchestrator)
                                .with_dt(dt)
                                .make_host_shared();
        system->update();
    }

    atlas::FluidHostPtr
    fluid() const {
        return _fluid;
    }

private:
    atlas::FluidHostPtr _fluid;
    atlas::UniverseHostPtr _universe;
    atlas::SearcherHostPtr _searcher;
    atlas::SolveHostPtr _solver;
    atlas::OrchestratorHostPtr _orchestrator;
    atlas::SourceHostPtr _source;
    atlas::SinkHostPtr _sink;
    atlas::ColliderHostPtr _collider;
    SolverSpec _spec;
};

}

NB_MODULE(atlas, m) {
    m.doc() = "Atlas Engine Python bindings";

    m.attr("__version__") = ATLAS_VERSION_STRING;

    m.def("backend", &atlas_backend,
          "Return the tasking backend the module was compiled against.");

    nb::class_<atlas::Float3>(m, "Float3")
        .def(nb::init<>())
        .def(nb::init<float, float, float>(), "x"_a, "y"_a, "z"_a)
        .def_rw("x", &atlas::Float3::x)
        .def_rw("y", &atlas::Float3::y)
        .def_rw("z", &atlas::Float3::z)
        .def("dot", &atlas::Float3::dot, "v"_a)
        .def("cross", &atlas::Float3::cross, "v"_a)
        .def("length", &atlas::Float3::length)
        .def("normalized", &atlas::Float3::normalized)
        .def(nb::self + nb::self)
        .def(nb::self - nb::self)
        .def(nb::self * float())
        .def(-nb::self)
        .def("__repr__", &vector3_repr);

    nb::class_<atlas::Int3>(m, "Int3")
        .def(nb::init<>())
        .def(nb::init<int, int, int>(), "x"_a, "y"_a, "z"_a)
        .def_rw("x", &atlas::Int3::x)
        .def_rw("y", &atlas::Int3::y)
        .def_rw("z", &atlas::Int3::z)
        .def("__repr__", &vector3i_repr);

    nb::class_<atlas::Quaternion>(m, "Quaternion")
        .def(nb::init<>())
        .def(nb::init<float, float, float, float>(), "w"_a, "x"_a, "y"_a, "z"_a)
        .def(nb::init<float, float, float>(), "rx"_a, "ry"_a, "rz"_a)
        .def_rw("w", &atlas::Quaternion::w)
        .def_rw("x", &atlas::Quaternion::x)
        .def_rw("y", &atlas::Quaternion::y)
        .def_rw("z", &atlas::Quaternion::z)
        .def("normalized", &atlas::Quaternion::normalized)
        .def("__repr__", &quaternion_repr);

    nb::class_<atlas::Float3x3>(m, "Float3x3")
        .def(nb::init<>())
        .def(nb::init<float, float, float, float, float, float, float, float, float>(),
             "m00"_a, "m01"_a, "m02"_a,
             "m10"_a, "m11"_a, "m12"_a,
             "m20"_a, "m21"_a, "m22"_a)
        .def_rw("m00", &atlas::Float3x3::m00)
        .def_rw("m01", &atlas::Float3x3::m01)
        .def_rw("m02", &atlas::Float3x3::m02)
        .def_rw("m10", &atlas::Float3x3::m10)
        .def_rw("m11", &atlas::Float3x3::m11)
        .def_rw("m12", &atlas::Float3x3::m12)
        .def_rw("m20", &atlas::Float3x3::m20)
        .def_rw("m21", &atlas::Float3x3::m21)
        .def_rw("m22", &atlas::Float3x3::m22);

    nb::enum_<atlas::MaterialType::Value>(m, "MaterialType")
        .value("molecule", atlas::MaterialType::molecule)
        .value("atom", atlas::MaterialType::atom)
        .value("ion", atlas::MaterialType::ion)
        .value("neutron", atlas::MaterialType::neutron)
        .value("solid", atlas::MaterialType::solid);

    nb::class_<atlas::MaterialProperties>(m, "MaterialProperties")
        .def(nb::init<>())
        .def_rw("type", &atlas::MaterialProperties::type)
        .def_rw("mass", &atlas::MaterialProperties::mass)
        .def_rw("molecular_mass", &atlas::MaterialProperties::molecular_mass)
        .def_rw("species_id", &atlas::MaterialProperties::species_id)
        .def_rw("reference_diameter", &atlas::MaterialProperties::reference_diameter)
        .def_rw("reference_temperature", &atlas::MaterialProperties::reference_temperature)
        .def_rw("viscosity_index", &atlas::MaterialProperties::viscosity_index);

    nb::enum_<atlas::DsmcKernelType>(m, "DsmcKernelType")
        .value("hard_sphere", atlas::DsmcKernelType::hard_sphere)
        .value("variable_hard_sphere", atlas::DsmcKernelType::variable_hard_sphere)
        .value("variable_soft_sphere", atlas::DsmcKernelType::variable_soft_sphere);

    nb::enum_<atlas::SphKernelType>(m, "SphKernelType")
        .value("standard", atlas::SphKernelType::standard)
        .value("cubic_spline", atlas::SphKernelType::cubic_spline)
        .value("wendland_quintic", atlas::SphKernelType::wendland_quintic);

    nb::enum_<atlas::SpawnType>(m, "SpawnType")
        .value("surface", atlas::SpawnType::surface)
        .value("volume", atlas::SpawnType::volume);

    nb::enum_<atlas::DespawnType>(m, "DespawnType")
        .value("surface", atlas::DespawnType::surface)
        .value("volume", atlas::DespawnType::volume)
        .value("tracing", atlas::DespawnType::tracing);

    nb::enum_<atlas::DiffuseSampling>(m, "DiffuseSampling")
        .value("cosine_weighted", atlas::DiffuseSampling::cosine_weighted)
        .value("uniform", atlas::DiffuseSampling::uniform);

    nb::class_<atlas::Generator>(m, "Generator");

    m.def("MaxwellBoltzmannGenerator", &make_maxwell_boltzmann_generator,
          "temperature"_a, "molecular_mass"_a, "bulk_velocity"_a, "seed"_a);

    nb::class_<atlas::Universe>(m, "Universe")
        .def(nb::new_(&make_universe),
             "lower_corner"_a, "upper_corner"_a, "cell_size"_a,
             "source_units"_a   = std::vector<atlas::Unit> {},
             "sink_units"_a     = std::vector<atlas::Unit> {},
             "collider_units"_a = std::vector<atlas::Unit> {})
        .def("cell_count", &atlas::Universe::cell_count);

    nb::class_<atlas::Fluid>(m, "Fluid")
        .def(nb::new_(&make_fluid), "buffer_size"_a, "properties"_a, "generators"_a)
        .def("particle_count", &atlas::Fluid::particle_count)
        .def("set_particle_count", &atlas::Fluid::set_particle_count, "particle_count"_a);

    nb::class_<atlas::Searcher>(m, "Searcher")
        .def("build", &atlas::Searcher::build);

    m.def("SpatialHashingSearcher", &make_spatial_hashing_searcher,
          "universe"_a, "fluid"_a);

    nb::class_<atlas::DsmcSolver>(m, "DsmcSolver")
        .def(nb::init<atlas::UniverseHostPtr, atlas::FluidHostPtr, atlas::SearcherHostPtr, atlas::DsmcKernelType>(),
             "universe"_a, "fluid"_a, "searcher"_a, "kernel_type"_a = atlas::DsmcKernelType::hard_sphere)
        .def("solve", [](atlas::DsmcSolver& s, float dt) { s.solve(dt); }, "dt"_a)
        .def("kernel_type", &atlas::DsmcSolver::kernel_type);

    nb::class_<atlas::SphSolver>(m, "SphSolver")
        .def(nb::init<atlas::UniverseHostPtr, atlas::FluidHostPtr, atlas::SearcherHostPtr, atlas::SphKernelType>(),
             "universe"_a, "fluid"_a, "searcher"_a, "kernel_type"_a = atlas::SphKernelType::standard)
        .def("solve", [](atlas::SphSolver& s, float dt) { s.solve(dt); }, "dt"_a)
        .def("kernel_type", &atlas::SphSolver::kernel_type);

    nb::class_<atlas::Geometry>(m, "Geometry");

    m.def("Box", &make_box, "lower_corner"_a, "upper_corner"_a);
    m.def("Sphere", &make_sphere, "center"_a, "radius"_a);
    m.def("Cylinder", &make_cylinder, "center"_a, "radius"_a, "height"_a, "open"_a = false);
    m.def("Circle", &make_circle, "center"_a, "normal"_a, "radius"_a);
    m.def("Plane", &make_plane, "point"_a, "normal"_a);
    m.def("Square", &make_square, "center"_a, "normal"_a, "side_length"_a);
    m.def("Triangle", &make_triangle, "a"_a, "b"_a, "c"_a);
    m.def("TriangleMesh", &make_triangle_mesh_from_obj, "filename"_a);

    nb::class_<atlas::SensorMetrics::Record>(m, "SensorRecord")
        .def_ro("step_index", &atlas::SensorMetrics::Record::step_index)
        .def_ro("unit_index", &atlas::SensorMetrics::Record::unit_index)
        .def_ro("particle_count", &atlas::SensorMetrics::Record::particle_count);

    nb::class_<atlas::Observer>(m, "Observer")
        .def(nb::new_(&make_observer), "reserve_count"_a = 0)
        .def("export_csv", &atlas::Observer::export_csv, "output_directory"_a)
        .def("source_records", &observer_source_records)
        .def("sink_records", &observer_sink_records);

    nb::class_<atlas::Sync>(m, "Sync")
        .def(nb::new_(&make_sync), "translation"_a, "orientation"_a);

    nb::class_<atlas::Unit>(m, "Unit")
        .def(nb::new_(&make_unit), "geometry"_a, "sync"_a);

    nb::class_<atlas::IsothermalSurfaceInteraction>(m, "IsothermalSurfaceInteraction")
        .def(nb::new_(&make_isothermal_interaction),
             "diffuse_sampling"_a = atlas::DiffuseSampling::cosine_weighted,
             "restitution"_a = 1.0f, "momentum_acc"_a = 1.0f, "temperature"_a = 300.0f);

    nb::enum_<atlas::MaxwellianInternalEnergyStyle>(m, "MaxwellianInternalEnergyStyle")
        .value("none", atlas::MaxwellianInternalEnergyStyle::none)
        .value("smooth", atlas::MaxwellianInternalEnergyStyle::smooth)
        .value("discrete", atlas::MaxwellianInternalEnergyStyle::discrete);

    nb::class_<atlas::MaxwellianSurfaceInteraction>(m, "MaxwellianSurfaceInteraction")
        .def(nb::new_(&make_maxwellian_interaction),
             "temperature"_a = 300.0f, "molecular_mass"_a = 4.651734e-26f,
             "momentum_acc"_a = 1.0f, "trans_acc"_a = 1.0f, "rot_acc"_a = 1.0f, "vib_acc"_a = 1.0f);

    nb::class_<atlas::Source>(m, "Source")
        .def(nb::new_(&make_source),
             "fluid"_a, "universe"_a, "spawn_type"_a = atlas::SpawnType::surface,
             "spacing"_a = 0.1f, "temperature"_a = 273.15f,
             "observer"_a = atlas::ObserverHostPtr {});

    nb::class_<atlas::Sink>(m, "Sink")
        .def(nb::new_(&make_sink),
             "fluid"_a, "universe"_a, "despawn_type"_a = atlas::DespawnType::volume, "flip"_a = false,
             "observer"_a = atlas::ObserverHostPtr {});

    nb::class_<atlas::Collider>(m, "Collider")
        .def(nb::new_(&make_collider),
             "fluid"_a, "universe"_a, "interaction"_a, "flip"_a = false)
        .def(nb::new_(&make_collider_maxwellian),
             "fluid"_a, "universe"_a, "interaction"_a, "flip"_a = false);

    nb::class_<SolverSpec>(m, "SolverSpec");

    m.def("Solver", [](const std::string& name) { return SolverSpec { name }; }, "name"_a,
          "Select a solver by name: 'dsmc' or 'sph'.");

    nb::class_<SimpleSystem>(m, "System")
        .def(nb::init<atlas::FluidHostPtr, atlas::UniverseHostPtr>(), "fluid"_a, "universe"_a)
        .def_prop_rw("solver", &SimpleSystem::solver, &SimpleSystem::set_solver)
        .def_prop_rw("source", &SimpleSystem::source, &SimpleSystem::set_source)
        .def_prop_rw("sink", &SimpleSystem::sink, &SimpleSystem::set_sink)
        .def_prop_rw("collider", &SimpleSystem::collider, &SimpleSystem::set_collider)
        .def("update", &SimpleSystem::update, "dt"_a)
        .def("fluid", &SimpleSystem::fluid);
}
