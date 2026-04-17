#pragma once

namespace atlas::system {

std::uint64_t
ShuffleOperator::shuffle_key(const int index,
                             const std::uint64_t seed) const noexcept {
    // Form the initial 64-bit input state from:
    // - the logical element index,
    // - the caller-provided seed,
    // - the operator's internal index offset.
    //
    // This lets the same index map to different shuffled keys under
    // different seeds or operator configurations.
    std::uint64_t value = static_cast<std::uint64_t>(index) + seed + index_offset;

    // First avalanche / mixing round:
    // - xor with a shifted copy to spread high/low bit influence,
    // - multiply by an odd-looking constant to further decorrelate bits.
    //
    // This is a standard integer-hash style transformation.
    value = (value ^ (value >> first_shift)) * first_multiplier;

    // Second avalanche / mixing round:
    // - repeat xor-shift scrambling,
    // - apply a second multiplicative mixer.
    //
    // Using a different shift and multiplier improves bit diffusion.
    value = (value ^ (value >> second_shift)) * second_multiplier;

    // Final whitening step:
    // - xor with a final shifted version of itself
    // to reduce remaining linear structure in the output bits.
    return value ^ (value >> final_shift);
}

std::uint64_t
ShuffleOperator::operator()(const int index,
                            const std::uint64_t seed) const noexcept {
    // Function-call convenience wrapper around shuffle_key().
    //
    // This allows ShuffleOperator to be used like a stateless hashing functor
    // in algorithms that expect callable objects.
    return shuffle_key(index, seed);
}

} // namespace atlas::system