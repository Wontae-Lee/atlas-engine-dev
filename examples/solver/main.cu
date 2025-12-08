#include <atlas/atlas.h>

using namespace atlas;

int
main() {

    Emitter<float> emitter { -8.0f, -7.0f, -1.0f, 1.0f, -7.0f, 8.0f, 0.05f, 0.5f, 0.0f };
    emitter.set_emit_per_step(10000);

    Remover<float> remover { -10.0f, 10.0f, -10.0f, 10.0f, -10.0f, 10.0f };
    Sphere<float> sphere { Vector3<float> { 0, 0, 0 }, 3.f };
    Collider<float> collider;
    collider.add_geometry(sphere);

    Advector<float> advector;
    advector.set_collider(collider);

    SpatialHashingSearcher<float> searcher;
    DSMCSolver<float> solver;
    auto solver_ptr   = atlas::make_host_shared<DSMCSolver<float>>(solver);

    ParticleSystem<float> psystem { 2000000 };
    psystem.set_emitter(emitter);
    psystem.set_remover(remover);
    psystem.set_advector(advector);
    psystem.set_solver(solver_ptr);

    for (int i = 0; i < 100; ++i) {
        psystem.update();
    }
    return 0;
}