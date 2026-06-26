#pragma once

#include <atlas/core/macros.h>
#include <atlas/parallel/parallel_for.h>
#include <atlas/sink/sink_probe.h>
#include <atlas/spatial/ray.h>

namespace atlas::detail {

template <typename T>
class SinkParticleDespawner final {
public:
    ATLAS_HOST ATLAS_FORCE_INLINE void
    apply(const SinkProbe<T>& probe, int* removed_unit_indices) const;
};

}

#include <atlas/sink/detail/sink_particle_despawner.hpp>