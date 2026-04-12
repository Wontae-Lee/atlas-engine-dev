#pragma once

#include <atlas/codec/codec.h>
#include <atlas/core/macros.h>
#include <atlas/domain/domain.h>
#include <atlas/fluid/fluid.h>
#include <atlas/memory/memory.h>
#include <atlas/searcher/spatial_hashing_searcher.h>
#include <atlas/solver/solver.h>

#include <stdexcept>
#include <utility>

namespace atlas::system {

template <typename T>
class Orchestrator final {
public:
    class Builder;

public:
    Orchestrator()  = default;
    ~Orchestrator() = default;

    ATLAS_HOST ATLAS_FORCE_INLINE explicit Orchestrator(SolveHostPtr<T> solver) noexcept;

    ATLAS_HOST ATLAS_FORCE_INLINE static Builder
    builder() noexcept;

    ATLAS_HOST ATLAS_FORCE_INLINE void
    solve(DomainDeviceProbe<T> domain,
          SpatialHashingProbe<T> searcher,
          FluidDeviceProbe<T> particle,
          CodecDeviceProbe<T> codec);

    ATLAS_HOST ATLAS_FORCE_INLINE void
    set_solver(SolveHostPtr<T> solver) noexcept;

    ATLAS_HOST ATLAS_NODISCARD ATLAS_FORCE_INLINE const SolveHostPtr<T>&
    solver() const noexcept;

private:
    SolveHostPtr<T> _solver {};
};

template <typename T>
class Orchestrator<T>::Builder final {
public:
    Builder() = default;

    ATLAS_HOST ATLAS_FORCE_INLINE Builder&
    with_solver(SolveHostPtr<T> solver) noexcept;

    ATLAS_HOST ATLAS_FORCE_INLINE Orchestrator<T>
    build() const;

    ATLAS_HOST ATLAS_FORCE_INLINE atlas::host_shared_ptr<Orchestrator<T>>
    make_host_shared() const;

private:
    ATLAS_HOST ATLAS_FORCE_INLINE void
    validate() const;

private:
    SolveHostPtr<T> _solver {};
};

}

namespace atlas {

template <typename T>
using Orchestrator = atlas::system::Orchestrator<T>;

template <typename T>
using OrchestratorHostPtr = atlas::host_shared_ptr<atlas::system::Orchestrator<T>>;

template <typename T>
using OrchestratorDevicePtr = atlas::device_shared_ptr<atlas::system::Orchestrator<T>>;

}

#include <atlas/orchestrator/orchestrator.hpp>
