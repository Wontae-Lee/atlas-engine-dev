#include <atlas/measure/measurer.h>

#include <atlas/fluid/fluid.h>
#include <atlas/searcher/spatial_hashing_searcher.h>
#include <atlas/universe/universe.h>

#include <gtest/gtest.h>

#include <utility>

namespace {

using atlas::Fluid;
using atlas::FluidHostPtr;
using atlas::MeasureModeType;
using atlas::Measurer;
using atlas::SearcherHostPtr;
using atlas::SpatialHashingSearcher;
using atlas::Universe;
using atlas::UniverseHostPtr;
using atlas::Vector3;

class DummyMeasure final : public Measurer {
public:
    DummyMeasure(UniverseHostPtr universe,
                 FluidHostPtr fluid,
                 SearcherHostPtr searcher,
                 MeasureModeType measure_mode) noexcept
        : Measurer(std::move(universe), std::move(fluid), std::move(searcher))
        , _measure_mode(measure_mode) { }

    using Measurer::measure;

    void
    measure() override {
        ++measure_calls;
    }

    ATLAS_NODISCARD MeasureModeType
    measure_mode() const noexcept override {
        return _measure_mode;
    }

    int measure_calls = 0;

private:
    MeasureModeType _measure_mode;
};

UniverseHostPtr
make_universe() {
    return Universe::builder()
        .with_lower_corner(Vector3(0.0f, 0.0f, 0.0f))
        .with_upper_corner(Vector3(1.0f, 1.0f, 1.0f))
        .with_cell_size(1.0f)
        .make_host_shared();
}

FluidHostPtr
make_fluid() {
    return Fluid::builder()
        .with_buffer_size(4)
        .make_host_shared();
}

SearcherHostPtr
make_searcher(const UniverseHostPtr& universe,
              const FluidHostPtr& fluid) {
    return SpatialHashingSearcher::builder()
        .with_universe(universe)
        .with_fluid(fluid)
        .make_host_shared();
}

}

TEST(Measure, DerivedImplementationStoresDependenciesAndMode) {
    const auto universe = make_universe();
    const auto fluid = make_fluid();
    const auto searcher = make_searcher(universe, fluid);

    DummyMeasure measure(universe, fluid, searcher, MeasureModeType::fluid);

    EXPECT_EQ(measure.measure_mode(), MeasureModeType::fluid);

    measure.measure();
    EXPECT_EQ(measure.measure_calls, 1);
}

TEST(Measure, MeasureWithDtForwardsToMeasure) {
    const auto universe = make_universe();
    const auto fluid = make_fluid();
    const auto searcher = make_searcher(universe, fluid);

    DummyMeasure measure(universe, fluid, searcher, MeasureModeType::all);

    measure.measure(0.5f);
    EXPECT_EQ(measure.measure_calls, 1);
}
