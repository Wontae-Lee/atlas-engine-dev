#include <atlas/solver/solver.h>

#include <gtest/gtest.h>

namespace {

using atlas::DeviceBuffer;
using atlas::Solver;

class DummySolver final : public Solver {
public:
    void
    solve(const float dt) override {
        last_dt = dt;
    }

    void
    solve(const DeviceBuffer<int>*, const int index, const float dt) override {
        last_index = index;
        last_dt    = dt;
    }

    float last_dt { 0.0f };
    int last_index { -1 };
};

}

TEST(Solver, VirtualSolveInterfacesAreOverridable) {
    DummySolver solver;

    solver.solve(0.25f);

    EXPECT_FLOAT_EQ(solver.last_dt, 0.25f);

    solver.solve(nullptr, 3, 0.5f);

    EXPECT_EQ(solver.last_index, 3);
    EXPECT_FLOAT_EQ(solver.last_dt, 0.5f);
}
