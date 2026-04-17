#pragma once

#include <atlas/buffer/host_buffer.h>
#include <atlas/codec/codec.h>
#include <atlas/core/macros.h>
#include <atlas/memory/memory.h>
#include <atlas/solver/solver.h>

namespace atlas::system {

template <typename T>
class Orchestrator final {
public:
    class Builder;

public:
    Orchestrator() = default;

    ~Orchestrator() = default;

    ATLAS_HOST ATLAS_FORCE_INLINE
    Orchestrator(CodecHostPtr<T> codec,
                 HostBuffer<SolveHostPtr<T>> solvers) noexcept;

    ATLAS_HOST ATLAS_FORCE_INLINE static Builder
    builder() noexcept;

    ATLAS_HOST ATLAS_FORCE_INLINE void
    orchestrate();

    ATLAS_HOST ATLAS_FORCE_INLINE void
    set_codec(CodecHostPtr<T> codec) noexcept;

    ATLAS_HOST ATLAS_FORCE_INLINE void
    add_solver(SolveHostPtr<T> solver) noexcept;

    ATLAS_HOST ATLAS_NODISCARD ATLAS_FORCE_INLINE const CodecHostPtr<T>&
    codec() const noexcept;

    ATLAS_HOST ATLAS_NODISCARD ATLAS_FORCE_INLINE const HostBuffer<SolveHostPtr<T>>&
    solvers() const noexcept;

private:
    CodecHostPtr<T> _codec {};

    HostBuffer<SolveHostPtr<T>> _solvers {};
};

template <typename T>
class Orchestrator<T>::Builder final {
public:
    Builder() = default;

    ATLAS_HOST ATLAS_FORCE_INLINE Builder&
    with_codec(CodecHostPtr<T> codec) noexcept;

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
    CodecHostPtr<T> _codec {};

    HostBuffer<SolveHostPtr<T>> _solvers {};
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
