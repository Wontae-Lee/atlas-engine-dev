#pragma once
#include <atlas/core/macros.h>

namespace atlas {

/**
 * @file device_pair_indexer.h
 * @brief Utility for mapping unordered pairs of indices to a unique linear index.
 *
 * @details
 * `DevicePairIndexer` provides a compact mapping from an **unordered pair**
 * `(s, r)` with `0 <= s, r < n_species` to a unique linear index.
 *
 * The mapping treats `(s, r)` and `(r, s)` as the **same pair** and is commonly
 * used for:
 * - Pairwise interaction tables
 * - Symmetric matrices stored in packed (upper-triangular) form
 * - Species–species interaction indexing
 *
 * Conceptually, the pairs are enumerated in row-major order over the
 * **upper triangular matrix including the diagonal**:
 *
 * \f[
 * (0,0),
 * (0,1), (1,1),
 * (0,2), (1,2), (2,2),
 * \ldots
 * \f]
 *
 * This allows storing \f$n(n+1)/2\f$ unique pairs instead of \f$n^2\f$.
 *
 * @tparam Int Integer type used for indices (e.g., `int`, `std::int32_t`).
 *
 * @note
 * - Designed to be callable from both host and device code.
 * - Assumes `0 <= s, r < n_species`; no bounds checking is performed.
 */

// ------------------------------------------------------------
// DevicePairIndexer
// ------------------------------------------------------------

/**
 * @brief Computes a unique linear index for an unordered pair `(s, r)`.
 *
 * @param s          First index.
 * @param r          Second index.
 * @param n_species  Total number of species (or categories).
 *
 * @return Linear index corresponding to the unordered pair `{s, r}`.
 *
 * @details
 * The function first enforces an ordering such that:
 * \f[
 *   s \le r
 * \f]
 * ensuring symmetry: `(s, r)` and `(r, s)` map to the same index.
 *
 * The returned index is computed as:
 * \f[
 *   \text{index} = s \cdot n - \frac{s(s-1)}{2} + (r - s)
 * \f]
 * where \f$n = \text{n\_species}\f$.
 *
 * This formula corresponds to flattening the upper triangular matrix
 * (including the diagonal) row by row.
 *
 * @example
 * For `n_species = 4`, the mapping is:
 * | (s,r) | index |
 * |-------|-------|
 * | (0,0) | 0 |
 * | (0,1) | 1 |
 * | (0,2) | 2 |
 * | (0,3) | 3 |
 * | (1,1) | 4 |
 * | (1,2) | 5 |
 * | (1,3) | 6 |
 * | (2,2) | 7 |
 * | (2,3) | 8 |
 * | (3,3) | 9 |
 *
 * @note
 * - The mapping is deterministic and collision-free for valid inputs.
 * - No runtime checks are performed for invalid indices.
 */
template <typename Int>
struct DevicePairIndexer {
    ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE Int
    pair_index(Int s, Int r, Int n_species) const {
        // Enforce ordering so that (s, r) and (r, s) map to the same index.
        if (r < s) {
            Int t = s;
            s     = r;
            r     = t;
        }

        // Compute linear index for the upper-triangular (including diagonal) layout.
        return s * n_species    // Offset to the start of row s
            - (s * (s - 1)) / 2 // Subtract the number of skipped lower-triangular elements
            + (r - s);          // Offset within row s
    }
};

} // namespace atlas
