#include <cuda/cuda_macros.cuh>
#include <utilities/tests_utils.cuh>

#include <atlas/atlas.h>

#include <stdexcept>

using namespace atlas;

CUDA_TEST(Measure, AliasHostPointerCanStoreDerivedMeasure) {
    auto measure = atlas::make_host_shared<atlas::test::DummyMeasure<double>>();
    atlas::MeasureHostPtr<double> base = measure;

    CUDA_ASSERT_NE(base, nullptr);
    CUDA_EXPECT_EQ(base->measure_mode(), atlas::system::MeasureModeType::All);
    base->measure({}, {}, {});
    CUDA_EXPECT_EQ(measure->call_count, 1);
}

CUDA_TEST(Measure, SystemBuilderAllowsMeasuresForIsothermalDomain) {
    const auto domain = atlas::test::make_isothermal_domain_ptr<double>(325.0);
    const auto measure = atlas::VarianceThermometer<double>::builder().make_host_shared();

    const auto sim_system = atlas::System<double>::builder()
                                .with_fluid(test::make_buffered_fluid<double>(4))
                                .with_domain(domain)
                                .with_measure(measure)
                                .build();

    CUDA_EXPECT_EQ(sim_system.domain(), domain);
    CUDA_EXPECT_EQ(sim_system.measure(), measure);
    CUDA_EXPECT_EQ(sim_system.measure()->measure_mode(), atlas::system::MeasureModeType::All);
}
