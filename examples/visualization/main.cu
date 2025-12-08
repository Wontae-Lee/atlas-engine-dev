
#include <atlas/atlas.h>
#include <vizkit/vizkit.h>
#include <memory>
#include <string>

using namespace atlas;
using namespace atlas::vizkit;

// ----------------------------
// main
// ----------------------------
int
main() {
    Emitter<float> emitter { -8.0f, -7.0f, -1.0f, 1.0f, -7.0f, 8.0f, 1000000.5f, 0.5f, 0.0f };
    emitter.set_emit_per_step(1000);

    Remover<float> remover { -10.0f, 10.0f, -10.0f, 10.0f, -10.0f, 10.0f };
    Sphere<float> sphere { Vector3<float> { 0, 0, 0 }, 3.f };
    Collider<float> collider;
    collider.add_geometry(sphere);

    Advector<float> advector;
    advector.set_collider(collider);

    SpatialHashingSearcher<float> searcher;
    DsmcSolver<float> solver;
    auto solver_ptr   = atlas::make_host_shared<DsmcSolver<float>>(solver);

    ParticleSystem<float> psystem { 2000000 };
    psystem.set_emitter(emitter);
    psystem.set_remover(remover);
    psystem.set_advector(advector);
    psystem.set_solver(solver_ptr);

    auto psystem_host_ptr = atlas::make_host_shared<ParticleSystem<float>>(psystem);
    auto particle_layer   = std::make_shared<ParticleLayer<float>>(psystem_host_ptr, 2000000);
    auto box_layer        = std::make_shared<BoxLayer<float>>(Vector3F { -10.0f, -10.0f, -10.0f }, Vector3F { 10.0f, 10.0f, 10.0f });
    auto sphere_layer     = std::make_shared<SphereLayer<float>>(Vector3<float> { 0, 0, 0 }, 3.f);

    Viewer<float> viewer(1280, 720, "Atlas CUDA Particles");
    viewer.add_layer(particle_layer);
    viewer.add_layer(box_layer);
    viewer.add_layer(sphere_layer);
    viewer.run();
}
