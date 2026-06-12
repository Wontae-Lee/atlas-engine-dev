#pragma once

#include <atlas/solver/dsmc/dsmc_probe.h>
#include <atlas/solver/sph/sph_probe.h>

#include <cstdint>

namespace atlas {

template <typename T>
struct HybridDsmcSphProbe {
    SphProbe<T> sph {};
    DsmcProbe<T> dsmc {};
    T grouping_length {};
    int sph_particle_threshold {};
    std::uint64_t collision_seed {};
    bool pairing_without_replacement {};
};

template <typename T>
using HybridProbe = HybridDsmcSphProbe<T>;

}

namespace atlas {
template <typename T>
using HybridProbe = atlas::HybridDsmcSphProbe<T>;

}
