#include <atlas/atlas.h>

using namespace atlas;

int
main() {
    using T = float;

    Emitter<T> emitter {
        -8.0f,
        -7.0f,
        -1.0f,
        1.0f,
        -7.0f,
        8.0f,
        1000000.5f,
        0.5f,
        0.0f
    };
    emitter.set_emit_per_step(1000);

    Remover<T> remover {
        -10.0f,
        10.0f,
        -10.0f,
        10.0f,
        -10.0f,
        10.0f
    };

    Sphere<T> sphere { Vector3<T> { 0, 0, 0 }, 3.0f };
    Collider<T> collider;
    collider.add_geometry(sphere);

    Advector<T> advector;
    advector.set_collider(collider);

    DsmcSolver<T> solver;

    auto emitter_ptr  = atlas::make_host_shared<Emitter<T>>(emitter);
    auto remover_ptr  = atlas::make_host_shared<Remover<T>>(remover);
    auto advector_ptr = atlas::make_host_shared<Advector<T>>(advector);
    auto solver_ptr   = atlas::make_host_shared<DsmcSolver<T>>(solver);

    ParticleSystem<T> system {
        2000000,
        emitter_ptr,
        remover_ptr,
        advector_ptr,
        solver_ptr
    };

    for (int step = 0; step < 1000; ++step) {
        system.update();
    }
}