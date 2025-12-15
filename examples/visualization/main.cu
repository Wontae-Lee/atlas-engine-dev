#include <atlas/atlas.h>
#include <memory>
#include <vizkit/vizkit.h>
#include <lyra/lyra.hpp>

using namespace atlas;
using namespace atlas::vizkit;

int
main(int argc, char** argv) {

    int width = 0;
    auto cli  = lyra::cli()
        | lyra::opt(width, "width")
        ["-w"]["--width"]("How wide should it be?");
    auto result = cli.parse({ argc, argv });
    if (!result) {
        std::cerr << "Error in command line: " << result.message() << std::endl;
        exit(1);
    }

    using sim_t = float;

    Emitter<sim_t> emitter{ -8.0f, -7.0f, -1.0f,
                            1.0f, -7.0f, 8.0f,
                            100000.5f, 0.5f, 0.0f };

    emitter.set_emit_per_step(100);

    Remover<sim_t> remover{
        -10.0f,
        10.0f,
        -10.0f,
        10.0f,
        -10.0f,
        10.0f
    };

    Sphere<sim_t> sphere{ Vector3<sim_t>{ 0, 0, 0 }, 3.0f };
    Collider<sim_t> collider;
    collider.add_geometry(sphere);

    Advector<sim_t> advector;
    advector.set_collider(collider);

    DsmcSolver<sim_t> solver;

    auto emitter_ptr  = atlas::make_host_shared<Emitter<sim_t>>(emitter);
    auto remover_ptr  = atlas::make_host_shared<Remover<sim_t>>(remover);
    auto advector_ptr = atlas::make_host_shared<Advector<sim_t>>(advector);
    auto solver_ptr   = atlas::make_host_shared<DsmcSolver<sim_t>>(solver);

    ParticleSystem<sim_t> system{
        0.00001f,
        emitter_ptr,
        remover_ptr,
        advector_ptr,
        solver_ptr,
        2000000
    };

    auto psystem_host_ptr = atlas::make_host_shared<ParticleSystem<float>>(system);
    auto particle_layer   = std::make_shared<ParticleLayer<float>>(psystem_host_ptr, 2000000);
    auto box_layer        = std::make_shared<BoxLayer<float>>(Vector3F{ -10.0f, -10.0f, -10.0f }, Vector3F{ 10.0f, 10.0f, 10.0f });
    auto sphere_layer     = std::make_shared<SphereLayer<float>>(Vector3<float>{ 0, 0, 0 }, 3.f);

    Viewer<float> viewer(1280, 720, "Atlas CUDA Particles");
    viewer.add_layer(particle_layer);
    viewer.add_layer(box_layer);
    viewer.add_layer(sphere_layer);
    viewer.run();
}