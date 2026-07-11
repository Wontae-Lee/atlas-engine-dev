#include <atlas/solver/solver_type.h>

#include <gtest/gtest.h>

#include <type_traits>

namespace {

using atlas::SolverType;

}

// The tag doubles as a serialized discriminant and a device-side comparison value, so its
// underlying type must be a fixed int.
static_assert(std::is_same_v<std::underlying_type_t<SolverType>, int>,
              "SolverType must be backed by int for host/device stability");

TEST(SolverType, DsmcIsTheFirstEnumerator) {
    // Only one solver kind exists today; pinning its value guards the serialization contract.
    EXPECT_EQ(static_cast<int>(SolverType::dsmc), 0);
}

TEST(SolverType, RoundTripsThroughUnderlyingInt) {
    const auto value = static_cast<int>(SolverType::dsmc);

    EXPECT_EQ(static_cast<SolverType>(value), SolverType::dsmc);
}
