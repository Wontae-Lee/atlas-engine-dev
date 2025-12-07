#include <atlas/atlas.h>

using namespace atlas;

int
main() {

    Emitter<float> emitter { -0.9f, -0.89f, -1.0f, 1.0f, -1.0f, 1.0f, 1, 0.1f, 10.0f };
    Remover<float> remover { -10.0f, 10.0f, -10.0f, 10.0f, -10.0f, 10.0f };
    Sphere<float> sphere { Vector3<float> { 0, 0, 0 }, 0.5f };

    auto sp_op = sphere.make_trace_operator();
    Advector<float> collider;
    collider.add_trace_operator(sp_op);
    auto collider_host_ptr = atlas::make_host_shared<Advector<float>>(collider);
    auto remover_host_ptr  = atlas::make_host_shared<Remover<float>>(remover);
    auto emitter_host_ptr  = atlas::make_host_shared<Emitter<float>>(emitter);

    ParticleSystem<float> psystem { 100000 };
    psystem.set_emitter(emitter_host_ptr);
    psystem.set_remover(remover_host_ptr);
    psystem.set_advector(collider_host_ptr);

    for (int i = 0; i < 10; ++i) {
        psystem.update();
    }
    return 0;
}