#include "../utilities/tests_utils.h"

#include <atlas/generator/generate_operator.h>
#include <atlas/measure/measurer.h>
#include <atlas/searcher/spatial_hashing_searcher.h>

#include <gtest/gtest.h>

namespace {

using T = float;

class DummyMeasure final : public atlas::system::Measurer<T> {
public:
    DummyMeasure(atlas::UniverseHostPtr<T> universe,
                 atlas::FluidHostPtr<T> fluid,
                 atlas::SpatialHashingSearcherHostPtr<T> searcher,
                 atlas::MeasureModeType measure_mode) noexcept
        : atlas::system::Measurer<T>(std::move(universe), std::move(fluid), std::move(searcher))
        , _measure_mode(measure_mode) {}

    void
    measure() override {
        ++measure_calls;
    }

    ATLAS_NODISCARD atlas::MeasureModeType
    measure_mode() const noexcept override {
        return _measure_mode;
    }

    int measure_calls = 0;

private:
    atlas::MeasureModeType _measure_mode;
};

atlas::UniverseHostPtr<T>
make_universe() {
    return atlas::universe::Universe<T>::builder()
        .with_lower_corner(atlas::Vector3<T>(0, 0, 0))
        .with_upper_corner(atlas::Vector3<T>(1, 1, 1))
        .with_cell_size(1.0f)
        .make_host_shared();
}

atlas::FluidHostPtr<T>
make_fluid() {
    return atlas::fluid::Fluid<T>::builder()
        .with_buffer_size(4)
        .make_host_shared();
}

atlas::SpatialHashingSearcherHostPtr<T>
make_searcher(const atlas::UniverseHostPtr<T>& universe,
              const atlas::FluidHostPtr<T>& fluid) {
    return atlas::system::SpatialHashingSearcher<T>::builder()
        .with_universe(universe)
        .with_fluid(fluid)
        .make_host_shared();
}

} // namespace

TEST(Measure, DerivedImplementationStoresDependenciesAndMode) {
    const auto universe = make_universe();
    const auto fluid = make_fluid();
    const auto searcher = make_searcher(universe, fluid);

    DummyMeasure measure(universe, fluid, searcher, atlas::MeasureModeType::Fluid);

    EXPECT_EQ(measure.measure_mode(), atlas::MeasureModeType::Fluid);

    measure.measure();
    EXPECT_EQ(measure.measure_calls, 1);
}
