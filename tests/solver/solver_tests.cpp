#include "../utilities/test_utils.h"

#include <atlas/solver/solver.h>

#include <testkit/testkit.h>

namespace {

using atlas::DeviceBuffer;
using atlas::Solver;

class DummySolver final : public Solver<float> {
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

    float last_dt { 0 };
    int last_index { -1 };
};

} // namespace

TEST(Solver, VirtualSolveInterfacesAreOverridable) {
    // Arrange: create a concrete test implementation of the solver interface.
    DummySolver solver;

    // Act: call the scalar solve overload.
    solver.solve(0.25f);

    // Assert: the scalar solve overload receives dt.
    EXPECT_FLOAT_EQ(solver.last_dt, 0.25f);

    // Act: call the indexed solve overload.
    solver.solve(nullptr, 3, 0.5f);

    // Assert: the indexed solve overload receives index and dt.
    EXPECT_EQ(solver.last_index, 3);
    EXPECT_FLOAT_EQ(solver.last_dt, 0.5f);
}
