#include <atlas/atlas.h>
#include <tbb/blocked_range.h>
#include <tbb/parallel_for.h>

#include <cmath>
#include <iostream>
#include <random>
#include <vector>

using namespace atlas;

static void
make_scene(std::vector<atlas::Triangle<float>>& tris) {
    using V3 = atlas::Vector3<float>;

    auto add = [&](const V3& a, const V3& b, const V3& c) {
        tris.emplace_back(a, b, c);
    };

    {
        float z = 0.0f;
        V3 p0 { -1.0f, -1.0f, z }, p1 { 1.0f, -1.0f, z }, p2 { 1.0f, 1.0f, z }, p3 { -1.0f, 1.0f, z };
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

    std::vector<TriangleF> triangles;
    make_scene(triangles);

    atlas::HostBuffer<TriangleF> h_tris;
    h_tris.resize(static_cast<int>(triangles.size()));
    for (int i = 0; i < static_cast<int>(triangles.size()); ++i)
        h_tris[i] = triangles[i];

    SAHBVH<float> bvh;
    bvh.set_leaf_size(16);
    bvh.set_num_of_bins(64);
    bvh.build(h_tris);

    constexpr int N        = 20000;
    constexpr int STEPS    = 300;
    constexpr float dt     = 0.003f;
    constexpr float bounce = 0.5f;
    const Vector3F gravity { 0.0f, 0.0f, -9.8f };

    DeviceBuffer<Vector3F> pos(N), vel(N);

    std::random_device rd;
    std::mt19937 rng(rd());
    std::uniform_real_distribution<float> Ux(-0.9f, 0.9f);
    std::uniform_real_distribution<float> Uy(-0.9f, 0.9f);
    std::uniform_real_distribution<float> Uz(0.1f, 0.8f);

    for (int i = 0; i < N; ++i) {
        pos[i] = Vector3F { Ux(rng), Uy(rng), Uz(rng) };
        vel[i] = Vector3F { 0.0f, 0.0f, 0.0f };
    }

    const Vector3F bmin { -1.0f, -1.0f, 0.0f };
    const Vector3F bmax { 1.0f, 1.0f, 1.2f };

    auto clamp_box = [&](float& x, float& vx, float lo, float hi) {
        if (x < lo) {
            x  = lo;
            vx = -vx * bounce;
        }
        if (x > hi) {
            x  = hi;
            vx = -vx * bounce;
        }
    };
    auto trace = bvh.make_trace_operator();
    for (int s = 0; s < STEPS; ++s) {
        tbb::parallel_for(tbb::blocked_range<int>(0, N, 2048),
                          [&](const tbb::blocked_range<int>& r) {
                              for (int i = r.begin(); i < r.end(); ++i) {
                                  vel[i]      = vel[i] + gravity * dt;
                                  Vector3F p0 = pos[i];
                                  Vector3F v  = vel[i];
                                  Vector3F p1 = p0 + v * dt;

                                  float px = p1.x, py = p1.y, pz = p1.z;
                                  float vx = v.x, vy = v.y, vz = v.z;
                                  clamp_box(px, vx, bmin.x, bmax.x);
                                  clamp_box(py, vy, bmin.y, bmax.y);
                                  clamp_box(pz, vz, bmin.z, bmax.z);
                                  p1 = Vector3F { px, py, pz };
                                  v  = Vector3F { vx, vy, vz };

                                  const Vector3F dir = p1 - p0;
                                  const float segL   = length(dir);

                                  if (segL > 1e-9f) {
                                      atlas::Ray<float> ray {};
                                      ray.origin    = p0;
                                      ray.direction = dir;

                                      auto h = trace(ray);
                                      if (h.is_intersecting && h.distance <= segL) {
                                          p1          = h.point + h.normal * eps;
                                          Vector3F rV = reflect(v, h.normal);
                                          v           = rV * bounce;
                                      }
                                  }

                                  vel[i] = v;
                                  pos[i] = p1;
                              }
                          });

    }

    for (int i = 0; i < std::min(10, N); ++i) {
        const auto& p = pos[i];
        std::cout << "p" << i << ": (" << p.x << ", " << p.y << ", " << p.z << ")\n";
    }

    return 0;
}