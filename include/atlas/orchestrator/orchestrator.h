#pragma once

/**
 * @file orchestrator.h
 * @brief Declares the Orchestrator class used to coordinate codec-aware solver execution.
 */

#include <atlas/buffer/host_buffer.h>
#include <atlas/codec/codec.h>
#include <atlas/core/macros.h>
#include <atlas/memory/memory.h>
#include <atlas/solver/solver.h>

namespace atlas::system {

/**
 * @brief Coordinates execution of a sequence of solvers, optionally with a codec.
 *
 * An Orchestrator stores:
 * - an optional codec,
 * - an ordered list of solver objects.
 *
 * Its primary responsibility is to invoke solvers in sequence according to the
 * current orchestration mode:
 * - if no codec is configured, each solver is invoked with solve()
 * - if a codec is configured, each solver is invoked with solve(allocated_solver)
 *
 * This allows the solver pipeline to adapt its behavior depending on whether
 * a codec-aware execution path is available.
 *
 * @tparam T Scalar type associated with the simulation system.
 */
template <typename T>
class Orchestrator final {
public:
    /**
     * @brief Builder for configuring and constructing Orchestrator instances.
     */
    class Builder;

public:
    /**
     * @brief Default constructor.
     */
    Orchestrator() = default;

    /**
     * @brief Destructor.
     */
    ~Orchestrator() = default;

    /**
     * @brief Constructs an orchestrator from a codec and solver list.
     *
     * @param codec Optional host-side shared pointer to a codec.
     * @param solvers Ordered list of host-side shared solver pointers.
     */
    ATLAS_HOST ATLAS_FORCE_INLINE
    Orchestrator(CodecHostPtr<T> codec,
                 HostBuffer<SolveHostPtr<T>> solvers) noexcept;

    /**
     * @brief Creates a Builder instance.
     *
     * @return Builder object for fluent orchestrator construction.
     */
    ATLAS_HOST ATLAS_FORCE_INLINE static Builder
    builder() noexcept;

    /**
     * @brief Executes the configured solver sequence.
     *
     * If no codec is present, each non-null solver is invoked with solve().
     * If a codec is present, each non-null solver is invoked with
     * solve(allocated_solver), where allocated_solver points to the codec-owned
     * device buffer.
     */
    ATLAS_HOST ATLAS_FORCE_INLINE void
    orchestrate();

    /**
     * @brief Sets or replaces the codec.
     *
     * @param codec Host-side shared pointer to the codec.
     */
    ATLAS_HOST ATLAS_FORCE_INLINE void
    set_codec(CodecHostPtr<T> codec) noexcept;

    /**
     * @brief Appends a solver to the orchestration list.
     *
     * @param solver Host-side shared pointer to the solver.
     */
    ATLAS_HOST ATLAS_FORCE_INLINE void
    add_solver(SolveHostPtr<T> solver) noexcept;

    /**
     * @brief Returns the configured codec.
     *
     * @return Const reference to the codec shared pointer.
     */
    ATLAS_HOST ATLAS_NODISCARD ATLAS_FORCE_INLINE const CodecHostPtr<T>&
    codec() const noexcept;

    /**
     * @brief Returns the configured solver list.
     *
     * @return Const reference to the solver container.
     */
    ATLAS_HOST ATLAS_NODISCARD ATLAS_FORCE_INLINE const HostBuffer<SolveHostPtr<T>>&
    solvers() const noexcept;

private:
    /**
     * @brief Optional codec used to control codec-aware solver execution.
     */
    CodecHostPtr<T> _codec {};

    /**
     * @brief Ordered list of solvers managed by this orchestrator.
     */
    HostBuffer<SolveHostPtr<T>> _solvers {};
};

/**
 * @brief Builder for Orchestrator.
 *
 * This builder collects:
 * - an optional codec,
 * - an ordered sequence of solvers.
 *
 * Validation currently ensures that no stored solver pointer is null.
 *
 * @tparam T Scalar type associated with the simulation system.
 */
template <typename T>
class Orchestrator<T>::Builder final {
public:
    /**
     * @brief Default constructor.
     */
    Builder() = default;

    /**
     * @brief Sets the codec used by the orchestrator.
     *
     * @param codec Host-side shared pointer to the codec.
     * @return Reference to this builder.
     */
    ATLAS_HOST ATLAS_FORCE_INLINE Builder&
    with_codec(CodecHostPtr<T> codec) noexcept;

    /**
     * @brief Appends a solver to the orchestrator configuration.
     *
     * @param solver Host-side shared pointer to the solver.
     * @return Reference to this builder.
     */
    ATLAS_HOST ATLAS_FORCE_INLINE Builder&
    with_solver(SolveHostPtr<T> solver) noexcept;

    /**
     * @brief Builds a validated Orchestrator object.
     *
     * @return Constructed Orchestrator object.
     *
     * @throw std::runtime_error Thrown if any configured solver is null.
     */
    ATLAS_HOST ATLAS_FORCE_INLINE Orchestrator<T>
    build() const;

    /**
     * @brief Builds a host-side shared Orchestrator object.
     *
     * @return Host shared pointer to a constructed Orchestrator object.
     *
     * @throw std::runtime_error Thrown if any configured solver is null.
     */
    ATLAS_HOST ATLAS_FORCE_INLINE atlas::host_shared_ptr<Orchestrator<T>>
    make_host_shared() const;

private:
    /**
     * @brief Validates the current builder state.
     *
     * @throw std::runtime_error Thrown if any solver pointer is null.
     */
    ATLAS_HOST ATLAS_FORCE_INLINE void
    validate() const;

private:
    /**
     * @brief Codec collected by the builder.
     */
    CodecHostPtr<T> _codec {};

    /**
     * @brief Solver list collected by the builder.
     */
    HostBuffer<SolveHostPtr<T>> _solvers {};
};

} // namespace atlas::system

namespace atlas {

/**
 * @brief Alias for atlas::system::Orchestrator.
 *
 * @tparam T Scalar type associated with the orchestrator.
 */
template <typename T>
using Orchestrator = atlas::system::Orchestrator<T>;

/**
 * @brief Host-side shared pointer alias for Orchestrator.
 *
 * @tparam T Scalar type associated with the orchestrator.
 */
template <typename T>
using OrchestratorHostPtr = atlas::host_shared_ptr<atlas::system::Orchestrator<T>>;

/**
 * @brief Device-side shared pointer alias for Orchestrator.
 *
 * @tparam T Scalar type associated with the orchestrator.
 */
template <typename T>
using OrchestratorDevicePtr = atlas::device_shared_ptr<atlas::system::Orchestrator<T>>;

} // namespace atlas

#include <atlas/orchestrator/orchestrator.hpp>
