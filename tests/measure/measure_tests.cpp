#include "../utilities/tests_utils.h"

#include <atlas/atlas.h>
#include <gtest/gtest.h>

#include <stdexcept>

using namespace atlas;

namespace {

template <typename T>
atlas::FluidHostPtr<T>
make_buffered_fluid(const std::size_t buffer_size) {
    return atlas::system::Fluid<T>::builder()
        .with_buffer_size(buffer_size)
        .make_host_shared();
}

}

TEST(Measure, AliasHostPointerCanStoreDerivedMeasure) {
    auto measure = atlas::make_host_shared<atlas::test::DummyMeasure<double>>();
    atlas::MeasureHostPtr<double> base = measure;

    ASSERT_NE(base, nullptr);
    base->measure({}, {}, {});
    EXPECT_EQ(measure->call_count, 1);
}

TEST(Measure, SystemBuilderRejectsMeasuresForIsothermalDomain) {
    const auto domain = atlas::test::make_isothermal_domain_ptr<double>(325.0);
    const auto measure = atlas::AverageThermometer<double>::builder().make_host_shared();

    EXPECT_THROW(
        (void)atlas::System<double>::builder()
            .with_fluid(make_buffered_fluid<double>(4))
            .with_domain(domain)
            .with_measure(measure)
            .build(),
        std::runtime_error);
}
