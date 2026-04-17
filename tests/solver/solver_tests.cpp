#include "../utilities/tests_utils.h"

#include <atlas/solver/solver.h>

#include <gtest/gtest.h>

namespace {

using T = float;

class DummySolver final : public atlas::system::Solver<T> {
public:
    void
    solve(const T dt) override {
        last_dt = dt;
    }

    void
    solve(const atlas::DeviceBuffer<int>*, const int index, const T dt) override {
        last_index = index;
        last_dt    = dt;
    }

    T last_dt { 0 };
    int last_index { -1 };
};

} // namespace

TEST(Solver, VirtualSolveInterfacesAreOverridable) {
    DummySolver solver;

    solver.solve(0.25f);
    EXPECT_FLOAT_EQ(solver.last_dt, 0.25f);

    solver.solve(nullptr, 3, 0.5f);
    EXPECT_EQ(solver.last_index, 3);
    EXPECT_FLOAT_EQ(solver.last_dt, 0.5f);
}
