#include "reference.h"

#include <atlas/atlas.h>

#include <charconv>
#include <cstddef>
#include <cstdio>
#include <exception>
#include <stdexcept>
#include <string_view>
#include <utility>

namespace {

std::size_t
parse_steps(int argc, char** argv) {
    if (argc > 2) throw std::invalid_argument("expected at most one step count");
    if (argc == 1) return 40;
    const std::string_view input(argv[1]);
    std::size_t steps = 0;
    const auto result = std::from_chars(input.data(), input.data() + input.size(), steps);
    if (input.empty() || result.ec != std::errc() || result.ptr != input.data() + input.size()) {
        throw std::invalid_argument("steps must be a nonnegative decimal integer within size_t range");
    }
    return steps;
}

atlas::System
build_system() {
    auto materials = atlas::MaterialDictionary::builder()
        .with_material(atlas::Material(atlas::Molecule(
            4.65e-26f, 0.0f, 0.0f, 0.0f, 4.17e-10f, 273.0f, 0.74f, 1.0f)))
        .make_host_shared();

    auto fluid = atlas::Fluid::builder()
        .with_buffer_size(1024)
        .with_particle_count(2)
        .with_statistical_weight(1.0f)
        .with_materials(materials)
        .with_position({ atlas::Float3(-0.5f, -0.1f, 0.0f), atlas::Float3(-0.5f, 0.1f, 0.0f) })
        .with_velocity({ atlas::Float3(10.0f, 0.0f, 0.0f), atlas::Float3(10.0f, 0.0f, 0.0f) })
        .with_species({ 0, 0 })
        .make_host_unique();

    auto universe = atlas::Universe::builder()
        .with_lower_corner(atlas::Float3(-1.0f))
        .with_upper_corner(atlas::Float3(1.0f))
        .with_cell_size(0.25f)
        .make_host_unique();

    return atlas::System::builder()
        .with_fluid(std::move(fluid))
        .with_universe(std::move(universe))
        .with_dt(1.0e-4f)
        .build();
}

int
run(const std::size_t steps) {
    auto system = build_system();
    for (std::size_t step = 0; step < steps; ++step) {
        system.update();
    }

    const auto particle_count = system.fluid()->particle_count();
    const auto& data = system.fluid()->state<atlas::FluidPositionState>()->data();
    const atlas::HostBuffer<atlas::Float3> positions(data.begin(), data.begin() + particle_count);
    std::printf("completed: steps=%zu particles=%zu dt=%.1es\n",
                system.step(), particle_count, system.dt());
    for (std::size_t index = 0; index < particle_count; ++index) {
        std::printf("particle[%zu] position=(%.6f, %.6f, %.6f)\n",
                    index, positions[index].x, positions[index].y, positions[index].z);
    }
    return 0;
}

}

int
main(int argc, char** argv) {
    if (argc == 2 && std::string_view(argv[1]) == "--help") {
        std::printf("Usage: %s [steps]\nDefault: 40 steps; 0 constructs without advancing.\n", argv[0]);
        return 0;
    }
    try {
        return run(parse_steps(argc, argv));
    } catch (const std::exception& error) {
        std::fprintf(stderr, "00_template: %s\nUsage: %s [steps]\n", error.what(), argv[0]);
        return 1;
    }
}
