#pragma once

/**
 * @file device_pair_indexer.h
 * @brief Declares a lightweight device-friendly indexer for unordered species pairs.
 *
 * @details
 * This header defines @ref atlas::DevicePairIndexer, a small utility object used
 * to map an unordered pair of integer species indices `(s, r)` into a unique
 * linear index.
 *
 * The mapping is designed for symmetric pair tables where:
 * - the pair `(s, r)` is treated as equivalent to `(r, s)`,
 * - only the upper-triangular portion of an `n_species x n_species` pair matrix
 *   is stored,
 * - a compact one-dimensional array is used instead of a full dense 2D matrix.
 *
 * ## Intended use
 * This indexing scheme is useful for data structures such as:
 * - species-pair interaction coefficients,
 * - pairwise collision tables,
 * - symmetric transport-property lookup arrays,
 * - any runtime storage where pair order is irrelevant.
 *
 * ## Symmetry handling
 * The implementation first normalizes the input pair so that:
 * - `s <= r`
 *
 * This guarantees that both `(s, r)` and `(r, s)` map to the same linear index.
 *
 * ## Storage convention
 * The returned index corresponds to a packed upper-triangular layout including
 * diagonal entries:
 * - `(0,0), (0,1), ..., (0,n-1), (1,1), (1,2), ...`
 *
 * The exact formula used is:
 * \f[
 * \mathrm{index}(s,r) =
 * s \cdot n_{\mathrm{species}}
 * - \frac{s(s-1)}{2}
 * + (r-s)
 * \f]
 * after enforcing \f$s \le r\f$.
 *
 * ## Host/device usage
 * The indexer is marked `ATLAS_ALL_DEVICE`, making it suitable for use in:
 * - host-side code,
 * - device kernels,
 * - backend-parallel utility logic.
 *
 * ---
 */

#include <atlas/core/macros.h>

namespace atlas {

/**
 * @brief Computes a packed linear index for an unordered pair of integer ids.
 *
 * @details
 * @ref DevicePairIndexer provides a lightweight stateless mapping from an
 * unordered pair `(s, r)` to a unique linear index in a compact upper-triangular
 * storage layout.
 *
 * The mapping treats the pair as symmetric:
 * - `(s, r)` and `(r, s)` produce the same result.
 *
 * This avoids storing both halves of a symmetric pair matrix and is therefore
 * especially useful for pairwise physical models where interaction coefficients
 * are order-independent.
 *
 * ## Template parameter
 * The integer type @p Int is expected to be suitable for:
 * - species identifiers,
 * - array indexing,
 * - arithmetic involved in the packed-index formula.
 *
 * Typical choices include `int`, `std::int32_t`, or `std::size_t`-compatible
 * integer types depending on the surrounding codebase.
 *
 * ---
 *
 * @tparam Int Integer type used for species ids and the resulting packed index.
 */
template <typename Int>
struct DevicePairIndexer {
    /**
     * @brief Return the packed linear index corresponding to an unordered pair.
     *
     * @details
     * This function maps the input pair `(s, r)` into a unique index in the
     * packed upper-triangular storage of an `n_species x n_species` symmetric
     * matrix.
     *
     * The algorithm proceeds as follows:
     * 1. If necessary, swap the inputs so that `s <= r`.
     * 2. Compute the offset of row `s` in the packed upper-triangular layout.
     * 3. Add the in-row offset `(r - s)`.
     *
     * This ensures:
     * - `pair_index(s, r, n_species) == pair_index(r, s, n_species)`
     * - diagonal pairs `(k, k)` are included,
     * - each unordered pair maps to exactly one index.
     *
     * ## Example ordering
     * For `n_species = 4`, the packed sequence is:
     * @code
     * (0,0) -> 0
     * (0,1) -> 1
     * (0,2) -> 2
     * (0,3) -> 3
     * (1,1) -> 4
     * (1,2) -> 5
     * (1,3) -> 6
     * (2,2) -> 7
     * (2,3) -> 8
     * (3,3) -> 9
     * @endcode
     *
     * @param s First species or pair identifier.
     * @param r Second species or pair identifier.
     * @param n_species Total number of species in the symmetric pair system.
     * @return Packed linear index for the unordered pair `(s, r)`.
     *
     * @note
     * This function assumes the inputs are valid indices relative to
     * `n_species`. Bounds checking is not performed here.
     */
    ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE Int
    pair_index(Int s, Int r, Int n_species) const {

        if (r < s) {
            Int t = s;
            s     = r;
            r     = t;
        }

        return s * n_species
            - (s * (s - 1)) / 2
            + (r - s);
    }
};

} // namespace atlas