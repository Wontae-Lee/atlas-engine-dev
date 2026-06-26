#include "../utilities/test_utils.h"

#include <atlas/generator/generate_operator.h>
#include <atlas/measure/measurer.h>
#include <atlas/searcher/spatial_hashing_searcher.h>

#include <testkit/testkit.h>

namespace {

using atlas::Fluid;
using atlas::FluidHostPtr;
using atlas::MeasureModeType;
using atlas::Measurer;
using atlas::SpatialHashingSearcher;
using atlas::SearcherHostPtr;
using atlas::Universe;
using atlas::UniverseHostPtr;
using atlas::Vector3F;

class DummyMeasure final : public Measurer<float> {
public:
    DummyMeasure(UniverseHostPtr<float> universe,
                 FluidHostPtr<float> fluid,
                 SearcherHostPtr<float> searcher,
                 MeasureModeType measure_mode) noexcept
        : Measurer<float>(std::move(universe), std::move(fluid), std::move(searcher))
        , _measure_mode(measure_mode) {}

    void
    measure() override {
        ++measure_calls;
    }

    ATLAS_ALL_DEVICE ATLAS_NODISCARD MeasureModeType
    measure_mode() const noexcept override {
        return _measure_mode;
    }

    int measure_calls = 0;

private:
    MeasureModeType _measure_mode;
};

UniverseHostPtr<float>
make_universe() {
    return Universe<float>::builder()
        .with_lower_corner(Vector3F(0, 0, 0))
        .with_upper_corner(Vector3F(1, 1, 1))
        .with_cell_size(1.0f)
        .make_host_shared();
}

FluidHostPtr<float>
make_fluid() {
    return Fluid<float>::builder()
        .with_buffer_size(4)
        .make_host_shared();
}

SearcherHostPtr<float>
make_searcher(const UniverseHostPtr<float>& universe,
              const FluidHostPtr<float>& fluid) {
    return SpatialHashingSearcher<float>::builder()
        .with_universe(universe)
        .with_fluid(fluid)
        .make_host_shared();
}

} // namespace

TEST(Measure, DerivedImplementationStoresDependenciesAndMode) {
    const auto universe = make_universe();
    const auto fluid = make_fluid();
    const auto searcher = make_searcher(universe, fluid);

    DummyMeasure measure(universe, fluid, searcher, MeasureModeType::Fluid);

    EXPECT_EQ(measure.measure_mode(), MeasureModeType::Fluid);

    measure.measure();
    EXPECT_EQ(measure.measure_calls, 1);
}
