#pragma once

namespace atlas::system {

std::uint64_t
ShuffleOperator::shuffle_key(const int index,
                             const std::uint64_t seed) const noexcept {

    std::uint64_t value = static_cast<std::uint64_t>(index) + seed + index_offset;
    value               = (value ^ (value >> first_shift)) * first_multiplier;
    value               = (value ^ (value >> second_shift)) * second_multiplier;
    return value ^ (value >> final_shift);
}

std::uint64_t
ShuffleOperator::operator()(const int index,
                            const std::uint64_t seed) const noexcept {
    return shuffle_key(index, seed);
}

} // namespace atlas::system
