#include <atlas/atlas.h>
#include <memory>
#include <vizkit/vizkit.h>

using namespace atlas;
using namespace atlas::vizkit;

int
main() {

    //  도메인 만들고, 파티클 데이타 유저 정의 해서 시뮬레이션 돌리고,
    //  vizkit 으로 시각화 하기
    using T = float;

    Emitter<T> emitter {
        -8.0f,
        -7.0f,
        -1.0f,
        1.0f,
        -7.0f,
        8.0f,
        100000.5f,
        0.5f,
        0.0f
    };
    emitter.set_emit_per_step(100);

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
        0.00001f,
        emitter_ptr,
        remover_ptr,
        advector_ptr,
        solver_ptr,
        2000000
    };

    auto psystem_host_ptr = atlas::make_host_shared<ParticleSystem<float>>(system);
    auto particle_layer   = std::make_shared<ParticleLayer<float>>(psystem_host_ptr, 2000000);
    auto box_layer        = std::make_shared<BoxLayer<float>>(Vector3F { -10.0f, -10.0f, -10.0f }, Vector3F { 10.0f, 10.0f, 10.0f });
    auto sphere_layer     = std::make_shared<SphereLayer<float>>(Vector3<float> { 0, 0, 0 }, 3.f);

    Viewer<float> viewer(1280, 720, "Atlas CUDA Particles");
    viewer.add_layer(particle_layer);
    viewer.add_layer(box_layer);
    viewer.add_layer(sphere_layer);
    viewer.run();
}