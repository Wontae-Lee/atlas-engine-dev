#pragma once
namespace atlas::system {
std::uint64_t
ShuffleOperator::shuffle_key(const int index,
                             const std::uint64_t seed) const noexcept {
    std::uint64_t value = static_cast<std::uint64_t>(index) + seed + atlas::seed::SHUFFLE_HASH_INDEX_OFFSET;
    value               = (value ^ (value >> atlas::seed::SHUFFLE_HASH_FIRST_SHIFT)) * atlas::seed::SHUFFLE_HASH_FIRST_MULTIPLIER;
    value               = (value ^ (value >> atlas::seed::SHUFFLE_HASH_SECOND_SHIFT)) * atlas::seed::SHUFFLE_HASH_SECOND_MULTIPLIER;
    return value ^ (value >> atlas::seed::SHUFFLE_HASH_FINAL_SHIFT);
}

std::uint64_t
ShuffleOperator::operator()(const int index,
                            const std::uint64_t seed) const noexcept {
    return shuffle_key(index, seed);
}

}
