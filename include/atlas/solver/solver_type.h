#pragma once

namespace atlas {

/**
 * @brief Runtime tag identifying the concrete @ref Solver leaf behind a base pointer.
 *
 * Returned by @ref Solver::type so callers holding a @ref SolverHostPtr can branch on
 * the dynamic solver kind without RTTI (the System step pipeline switches on it). Each
 * enumerator corresponds to one final @ref Solver subclass. The underlying type is fixed
 * to @c int so the tag round-trips through serialization and device-side comparisons.
 */
enum class SolverType : int {

    dsmc ///< Direct Simulation Monte Carlo collision solver (@ref DsmcSolver).
};

}