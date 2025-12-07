#include <algorithm>
#include <atlas/atlas.h>
#include <cstdio>
#include <random>
#include <vector>

#include <thrust/for_each.h>
#include <thrust/iterator/constant_iterator.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/transform.h>

#include <cuda_runtime.h>

using namespace atlas;

static void
make_scene(std::vector<atlas::Triangle<float>>& tris) {
    using V3 = atlas::Vector3<float>;

    auto add = [&](const V3& a, const V3& b, const V3& c) {
        tris.emplace_back(a, b, c);
    };

    {
        float z = 0.0f;
        V3 p0 { -1.0f, -1.0f, z }, p1 { 1.0f, -1.0f, z };
        V3 p2 { 1.0f, 1.0f, z }, p3 { -1.0f, 1.0f, z };
        add(p0, p1, p2);
        add(p0, p2, p3);
    }

    {
        V3 c { 0.0f, 0.0f, 0.5f };
        float s = 0.35f;
        V3 v0   = c + V3 { 0.0f, 0.0f, s };
        V3 v1   = c + V3 { s, 0.0f, -s * 0.4f };
        V3 v2   = c + V3 { -s, 0.0f, -s * 0.4f };
        V3 v3   = c + V3 { 0.0f, s, -s * 0.4f };
        add(v0, v1, v2);
        add(v0, v2, v3);
        add(v0, v3, v1);
        add(v1, v3, v2);
    }
}

int
main() {

    int dev = 0;
    CUDA_CHECK(cudaGetDevice(&dev));
    cudaDeviceProp prop {};
    CUDA_CHECK(cudaGetDeviceProperties(&prop, dev));
    std::printf("Using GPU %d: %s (Compute Capability %d.%d)\n",
                dev,
                prop.name,
                prop.major,
                prop.minor);

    std::vector<TriangleF> triangles;
    make_scene(triangles);

    atlas::HostBuffer<TriangleF> h_tris;
    h_tris.resize(static_cast<int>(triangles.size()));
    for (int i = 0; i < static_cast<int>(triangles.size()); ++i)
        h_tris[i] = triangles[i];

    SAHBVH<float> bvh;
    bvh.build(h_tris);

    constexpr int N        = 20000;
    constexpr int STEPS    = 300;
    constexpr float bounce = 0.5f;
    constexpr float eps    = 1e-4f;
    const Vector3F gravity { 0.0f, 0.0f, -9.8f };

    std::random_device rd;
    std::mt19937 rng(rd());
    std::uniform_real_distribution<float> Ux(-0.9f, 0.9f);
    std::uniform_real_distribution<float> Uy(-0.9f, 0.9f);
    std::uniform_real_distribution<float> Uz(0.1f, 0.8f);

    HostBuffer<Vector3F> h_pos(N), h_vel(N);
    for (int i = 0; i < N; ++i) {
        h_pos[i] = Vector3F { Ux(rng), Uy(rng), Uz(rng) };
        h_vel[i] = Vector3F { 0.0f, 0.0f, 0.0f };
    }

    DeviceBuffer<Vector3F> pos(N), vel(N);
    thrust::copy(h_pos.begin(), h_pos.end(), pos.data());
    thrust::copy(h_vel.begin(), h_vel.end(), vel.data());

    auto d_pos = thrust::raw_pointer_cast(pos.data());
    auto d_vel = thrust::raw_pointer_cast(vel.data());
    auto trace = bvh.make_trace_operator();

    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    CUDA_CHECK(cudaEventRecord(start));

    for (int step = 0; step < STEPS; ++step) {
        constexpr float dt = 0.003f;

        thrust::transform(thrust::device,
                          vel.begin(),
                          vel.end(),
                          thrust::make_constant_iterator(gravity),
                          vel.begin(),
                          [dt] __device__(const Vector3F& v, const Vector3F& g) {
                              return v + g * dt;
                          });

        thrust::transform(thrust::device,
                          pos.begin(),
                          pos.end(),
                          vel.begin(),
                          pos.begin(),
                          [dt] __device__(const Vector3F& p, const Vector3F& v) {
                              return p + v * dt;
                          });

        thrust::for_each(thrust::device,
                         thrust::make_counting_iterator(0),
                         thrust::make_counting_iterator(N),
                         [=] __device__(int i) {
                             Vector3F p0 = d_pos[i];
                             Vector3F v  = d_vel[i] + gravity * dt;

                             Vector3F dir = v * dt;
                             float segL   = length(dir);

                             Vector3F p1 = p0;
                             if (segL > 1e-9f) {

                                 RayF ray { p0, dir };
                                 auto h = trace(ray);
                                 if (h.is_intersecting && h.distance <= segL) {

                                     p1          = h.point + h.normal * eps;
                                     Vector3F rV = reflect(v, h.normal);
                                     v           = rV * bounce;
                                 } else {
                                     p1 = p0 + dir;
                                 }
                             }

                             d_vel[i] = v;
                             d_pos[i] = p1;
                         });

        CUDA_CHECK(cudaDeviceSynchronize());
    }

    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));
    float ms = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
    std::printf("GPU simulation total time: %.3f ms (avg %.3f ms / step for %d steps)\n",
                ms,
                ms / STEPS,
                STEPS);

    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));

    HostBuffer<Vector3F> h_pos_out = pos;
    for (int i = 0; i < std::min(10, N); ++i) {
        const auto& p = h_pos_out[i];
        std::printf("p%02d: (%.6f, %.6f, %.6f)\n", i, p.x, p.y, p.z);
    }

    return 0;
}