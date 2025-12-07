#ifndef INCLUDE_ATLAS_RANDOM_RANDOM_H
#define INCLUDE_ATLAS_RANDOM_RANDOM_H

#ifdef ATLAS_USE_RANDOM
#define RNG_MULT (1664525u ^ (__TIME__[1] * 2654435761u))
#define RNG_ADD (1013904223u ^ (__TIME__[2] * 1234567u))

#define RANDOM_CONST1 (0xED5AD4BBu ^ (__LINE__ * 2654435761u))
#define RANDOM_CONST2 (0xAC4C1B51u ^ (__COUNTER__ * 1013904223u))
#define RANDOM_CONST3 (0x31848BABu ^ (__TIME__[0] * 1315423911u))
#include <atlas/math/math.h>

namespace atlas::random {
ATLAS_DEVICE ATLAS_FORCE_INLINE float
rng(std::uint32_t& s) {
    s = RNG_MULT * s + RNG_ADD;
    return ((s >> 8) & 0x00FFFFFF) / 16777216.0f;
}

ATLAS_DEVICE ATLAS_FORCE_INLINE uint32_t
hash_u32(uint32_t x) {
    x ^= x >> 17;
    x *= RANDOM_CONST1;
    x ^= x >> 11;
    x *= RANDOM_CONST2;
    x ^= x >> 15;
    x *= RANDOM_CONST3;
    x ^= x >> 14;
    return x;
}

ATLAS_DEVICE ATLAS_FORCE_INLINE float
rand01(const Vector3<float>& p) {

    uint32_t seed = 0;

    auto mix = [&](float v) {
        uint32_t u = *reinterpret_cast<uint32_t const*>(&v);
        seed ^= u + 0x9E3779B9u + (seed << 6) + (seed >> 2);
    };

    mix(p.x);
    mix(p.y);
    mix(p.z);

    seed = hash_u32(seed);

    float r = rng(seed);

    return r;
}

template <typename T>
ATLAS_DEVICE ATLAS_FORCE_INLINE T
rand01(const Vector3<T>& p) {
    float r = rand01(Vector3<float> { float(p.x), float(p.y), float(p.z) });
    return T(r);
}

}
#else
#include <atlas/math/math.h>

namespace atlas::random {
ATLAS_DEVICE ATLAS_FORCE_INLINE float
rng(uint32_t& s) {
    s = 1664525u * s + 1013904223u;
    return static_cast<float>((s >> 8) & 0x00FFFFFF) / 16777216.0f;
}
ATLAS_DEVICE ATLAS_FORCE_INLINE uint32_t
hash_u32(uint32_t x) {
    x ^= x >> 17;
    x *= 0xED5AD4BBu;
    x ^= x >> 11;
    x *= 0xAC4C1B51u;
    x ^= x >> 15;
    x *= 0x31848BABu;
    x ^= x >> 14;
    return x;
}

template <typename T>
ATLAS_DEVICE ATLAS_FORCE_INLINE T
rand01(const Vector3<T>& p) {
    const Vector3<T> c(T(12.9898), T(78.233), T(37.719));
    T v = math::dot(p, c);
    T s = std::sin(v) * T(43758.5453);
    return s - std::floor(s);
}

}
#endif

#endif