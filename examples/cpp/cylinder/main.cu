/**
 * @file main.cu
 * @brief Rarefied crossflow over a cylinder, driven straight through the Atlas API.
 *
 * A standalone DSMC example: nitrogen streams along +z past the cylinder mesh in
 * `assets/cylinder.obj`, which lies along the x axis. No harness — it builds a
 * System, steps it, and prints wall-clock timings.
 *
 * Usage:
 *   atlas_example_cylinder [steps] [assets_dir] [output_dir]
 */

#include <atlas/atlas.h>

#include <chrono>
#include <cstdio>
#include <cmath>
#include <cstdlib>
#include <filesystem>
#include <string>

namespace {

using atlas::Float3;

/// Nitrogen (N2) as a VHS/VSS species. Mass in kg, reference diameter in m.
constexpr float N2_MASS                 = 4.65e-26f;
constexpr float N2_REFERENCE_DIAMETER   = 4.17e-10f;
constexpr float N2_REFERENCE_TEMPERATURE = 273.0f;
constexpr float N2_VISCOSITY_INDEX      = 0.74f;
constexpr float N2_SCATTERING_PARAMETER = 1.0f;

/// Freestream: 1000 m/s along +z at 300 K.
constexpr float FREESTREAM_SPEED = 1000.0f;
constexpr float TEMPERATURE      = 300.0f;

/**
 * Each simulated particle stands for this many real molecules.
 *
 * Chosen so the freestream lands at Kn = lambda / D ~ 0.05, the transitional
 * regime DSMC exists for. The lattice source emits one particle per cell face
 * per step and a particle crosses a cell in CELL_SIZE / FREESTREAM_SPEED
 * seconds, so the steady-state occupancy is (CELL_SIZE / FREESTREAM_SPEED) / DT
 * = 25 particles per cell. Number density is then 25 * W / cell_volume, and the
 * hard-sphere mean free path 1 / (sqrt(2) * pi * d^2 * n) gives W = 3.236e16.
 *
 * A weight that is too small pushes Kn into the hundreds: the flow goes free
 * molecular, no collision is ever accepted, and the case silently degenerates
 * into a pure advection benchmark.
 */
constexpr float STATISTICAL_WEIGHT = 3.236e16f;

/// The cylinder spans x in [0, 10] with radius 0.25, so the domain wraps it with
/// margin only across the flow. Flow runs the full z extent.
constexpr float CELL_SIZE = 0.25f;

const Float3 DOMAIN_LOWER { -0.5f, -2.0f, -2.0f };
const Float3 DOMAIN_UPPER { 10.5f, 2.0f, 2.0f };

/// A slab thinner than one step's travel would let particles tunnel through it.
constexpr float SLAB_THICKNESS = 0.4f;

/// 4 m of z at 1000 m/s is 4 ms; one step moves a thermal molecule ~4 mm.
constexpr float DT = 1.0e-5f;

constexpr std::size_t BUFFER_SIZE     = 2'000'000;
constexpr std::size_t OBSERVE_INTERVAL = 50;

atlas::SyncHostPtr
identity_sync() {
    return atlas::Sync::builder()
        .with_rigid_pose(Float3(0.0f, 0.0f, 0.0f), atlas::Quaternion(1.0f, 0.0f, 0.0f, 0.0f))
        .make_host_shared();
}

atlas::Geometry
box_geometry(const Float3& lower, const Float3& upper) {
    return atlas::Geometry(atlas::Box::builder()
                               .with_lower_corner(lower)
                               .with_upper_corner(upper)
                               .build());
}

atlas::Unit
static_unit(const atlas::Geometry& geometry) {
    return atlas::Unit::builder()
        .with_geometry(geometry)
        .with_sync(identity_sync())
        .build();
}

/// A thin box spanning the domain in x and y, placed at a given z slab.
atlas::Unit
z_slab(const float z_lower, const float z_upper) {
    return static_unit(box_geometry(Float3(DOMAIN_LOWER.x, DOMAIN_LOWER.y, z_lower),
                                    Float3(DOMAIN_UPPER.x, DOMAIN_UPPER.y, z_upper)));
}

/// The same slab inset by one slab thickness in x and y, so it does not overlap
/// the four side sinks — a spawn inside a sink is despawned on the step it is born.
atlas::Unit
inset_z_slab(const float z_lower, const float z_upper) {
    return static_unit(
        box_geometry(Float3(DOMAIN_LOWER.x + SLAB_THICKNESS, DOMAIN_LOWER.y + SLAB_THICKNESS, z_lower),
                     Float3(DOMAIN_UPPER.x - SLAB_THICKNESS, DOMAIN_UPPER.y - SLAB_THICKNESS, z_upper)));
}

}

int
main(int argc, char** argv) {
    const std::size_t steps = (argc > 1) ? std::strtoul(argv[1], nullptr, 10) : 200;

    const std::filesystem::path assets = (argc > 2) ? argv[2] : ATLAS_EXAMPLE_ASSETS_DIR;
    const std::filesystem::path output = (argc > 3) ? argv[3] : "cylinder_out";

    const std::filesystem::path mesh_path = assets / "cylinder.obj";

    if (!std::filesystem::exists(mesh_path)) {
        std::fprintf(stderr, "cylinder benchmark: cannot find %s\n", mesh_path.c_str());
        return 1;
    }

    // The mesh owns the device buffers that its Geometry view points into, so it
    // must outlive the System that captures that view.
    auto cylinder_mesh = atlas::TriangleMesh::builder()
                             .load_from_obj(mesh_path.string())
                             .make_host_shared();

    auto materials = atlas::MaterialDictionary::builder()
                         .with_material(atlas::Material(atlas::Molecule(N2_MASS,
                                                                        0.0f,
                                                                        0.0f,
                                                                        0.0f,
                                                                        N2_REFERENCE_DIAMETER,
                                                                        N2_REFERENCE_TEMPERATURE,
                                                                        N2_VISCOSITY_INDEX,
                                                                        N2_SCATTERING_PARAMETER)))
                         .make_host_shared();

    auto fluid = atlas::Fluid::builder()
                     .with_buffer_size(BUFFER_SIZE)
                     .with_particle_count(0)
                     .with_statistical_weight(STATISTICAL_WEIGHT)
                     .with_materials(materials)
                     .make_host_unique();

    auto universe = atlas::Universe::builder()
                        .with_lower_corner(DOMAIN_LOWER)
                        .with_upper_corner(DOMAIN_UPPER)
                        .with_cell_size(CELL_SIZE)
                        .make_host_unique();

    const int cell_count = universe->cell_count();

    // Inflow slab at the -z face; the generator gives every spawned molecule the
    // freestream drift on top of its Maxwellian thermal velocity.
    auto inflow = atlas::make_host_shared<atlas::Source>(
        atlas::Source(atlas::VolumeSource::builder()
                          .with_unit(inset_z_slab(DOMAIN_LOWER.z, DOMAIN_LOWER.z + SLAB_THICKNESS))
                          .with_spacing(CELL_SIZE)
                          .build()));

    auto maxwellian = atlas::make_host_shared<atlas::Generator>(
        atlas::Generator(atlas::MaxwellBoltzmannGenerator::builder()
                             .with_species_ratios({ 1.0f })
                             .with_species_numbers({ 0.0f })
                             .with_material_dictionary(*materials)
                             .with_temperature(TEMPERATURE)
                             .with_bulk_velocity(Float3(0.0f, 0.0f, FREESTREAM_SPEED))
                             .with_seed(20260710u)
                             .build()));

    // A fully diffuse, thermalizing wall.
    const atlas::Collider cylinder(
        atlas::IsothermalCollider::builder()
            .with_unit(static_unit(cylinder_mesh->make_device_geometry_view()))
            .with_momentum_accommodation_coefficient(1.0f)
            .with_restitution(1.0f)
            .with_diffuse_sampling(atlas::DiffuseSampling::cosine_weighted)
            .build());

    auto solver = atlas::DsmcSolver::builder()
                      .with_kernel_type(atlas::DsmcKernelType::variable_hard_sphere)
                      .make_host_shared();

    auto observer = atlas::Observer::builder()
                        .with_interval(OBSERVE_INTERVAL)
                        .with_output_directory(output)
                        .make_host_shared();

    auto builder = atlas::System::builder();

    builder.with_fluid(std::move(fluid))
        .with_universe(std::move(universe))
        .with_solver(solver)
        .with_emitter(inflow, maxwellian)
        .with_collider(cylinder)
        .with_observer(observer)
        .with_dt(DT);

    // Six slabs, one per face: anything that leaves the domain is despawned on the
    // step it crosses out. Without them particles stream away forever and every
    // cell lookup outside the grid is wasted work.
    builder.with_sink(atlas::Sink(atlas::VolumeSink::builder()
                                      .with_unit(z_slab(DOMAIN_UPPER.z - SLAB_THICKNESS, DOMAIN_UPPER.z))
                                      .build()));

    const Float3 faces[4][2] = {
        { Float3(DOMAIN_LOWER.x, DOMAIN_LOWER.y, DOMAIN_LOWER.z),
          Float3(DOMAIN_LOWER.x + SLAB_THICKNESS, DOMAIN_UPPER.y, DOMAIN_UPPER.z) },
        { Float3(DOMAIN_UPPER.x - SLAB_THICKNESS, DOMAIN_LOWER.y, DOMAIN_LOWER.z),
          Float3(DOMAIN_UPPER.x, DOMAIN_UPPER.y, DOMAIN_UPPER.z) },
        { Float3(DOMAIN_LOWER.x, DOMAIN_LOWER.y, DOMAIN_LOWER.z),
          Float3(DOMAIN_UPPER.x, DOMAIN_LOWER.y + SLAB_THICKNESS, DOMAIN_UPPER.z) },
        { Float3(DOMAIN_LOWER.x, DOMAIN_UPPER.y - SLAB_THICKNESS, DOMAIN_LOWER.z),
          Float3(DOMAIN_UPPER.x, DOMAIN_UPPER.y, DOMAIN_UPPER.z) },
    };

    for (const auto& face : faces) {
        builder.with_sink(atlas::Sink(
            atlas::VolumeSink::builder().with_unit(static_unit(box_geometry(face[0], face[1]))).build()));
    }

    auto system = builder.build();

    // Report the regime the case actually lands in, so a change to the weight, the
    // cell size, or dt cannot quietly turn this into a free-molecular run.
    const double occupancy   = (CELL_SIZE / FREESTREAM_SPEED) / DT;
    const double cell_volume = CELL_SIZE * CELL_SIZE * CELL_SIZE;
    const double density     = occupancy * STATISTICAL_WEIGHT / cell_volume;
    const double mean_free_path
        = 1.0 / (std::sqrt(2.0) * atlas::pi * N2_REFERENCE_DIAMETER * N2_REFERENCE_DIAMETER * density);

    std::printf("cylinder: %d cells, dt=%.1e s, %zu steps\n", cell_count, DT, steps);
    std::printf("freestream: n=%.3e m^-3, lambda=%.4f m, Kn=%.3f (D=0.5 m), ~%.0f particles/cell\n",
                density,
                mean_free_path,
                mean_free_path / 0.5,
                occupancy);
    std::printf("%6s %12s %14s %10s\n", "step", "particles", "step_ms", "total_s");

    const auto start = std::chrono::steady_clock::now();

    for (std::size_t step = 0; step < steps; ++step) {
        const auto step_start = std::chrono::steady_clock::now();

        system.update();

        const auto now = std::chrono::steady_clock::now();

        const double step_ms = std::chrono::duration<double, std::milli>(now - step_start).count();
        const double total_s = std::chrono::duration<double>(now - start).count();

        if (step % 10 == 0 || step + 1 == steps) {
            std::printf("%6zu %12zu %14.3f %10.3f\n",
                        system.step(),
                        system.fluid()->particle_count(),
                        step_ms,
                        total_s);
        }
    }

    const double elapsed = std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();

    std::printf("\n%zu steps in %.3f s (%.3f ms/step), %zu particles alive\n",
                steps,
                elapsed,
                1000.0 * elapsed / static_cast<double>(steps),
                system.fluid()->particle_count());
    std::printf("csv written to %s/data\n", output.c_str());

    return 0;
}
