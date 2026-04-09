#include "../../utilities/tests_utils.h"

#include <atlas/atlas.h>
#include <gtest/gtest.h>

#include <array>
#include <stdexcept>

using namespace atlas;

TEST(RmsThermometer, BuilderDefaultsToRmsType) {
    auto thermometer = atlas::RmsThermometer<double>::builder().build();

    EXPECT_EQ(thermometer.type(), atlas::system::ThermometerType::Rms);
    EXPECT_TRUE(thermometer.is_valid());
}

TEST(RmsThermometer, BuilderMakeHostSharedCreatesValidPointer) {
    const auto thermometer = atlas::RmsThermometer<double>::builder().make_host_shared();

    ASSERT_NE(thermometer, nullptr);
    EXPECT_EQ(thermometer->type(), atlas::system::ThermometerType::Rms);
}

TEST(RmsThermometer, MeasureThrowsForIsothermalDomain) {
    const auto domain = atlas::test::make_isothermal_domain_ptr<double>(300.0);
    auto searcher     = atlas::test::make_single_range_searcher(domain);
    atlas::system::Fluid<double> particle_data(2);

    auto domain_probe   = domain->make_device_probe();
    auto search_probe   = searcher.make_device_probe();
    auto particle_probe = particle_data.make_device_probe();

    auto thermometer = atlas::RmsThermometer<double>::builder().build();

    EXPECT_THROW(thermometer.measure(domain_probe, search_probe, particle_probe), std::runtime_error);
}

TEST(RmsThermometer, OperatorComputesCellTemperatureAndWritesParticleTemperature) {
    const auto domain = atlas::Domain<double>::builder()
                            .with_lower_corner(Vector3<double>(0.0, 0.0, 0.0))
                            .with_upper_corner(Vector3<double>(1.0, 1.0, 1.0))
                            .with_cell_size(0.5)
                            .make_host_shared();
    auto searcher = atlas::test::make_single_range_searcher(domain);
    atlas::system::Fluid<double> particle_data(3);
    auto particle_probe           = particle_data.make_device_probe();
    particle_probe.particle_count = 3;

    const std::array<Vector3<double>, 3> positions {
        Vector3<double>(0.10, 0.10, 0.10),
        Vector3<double>(0.20, 0.20, 0.20),
        Vector3<double>(0.80, 0.80, 0.80)
    };
    const std::array<Vector3<double>, 3> velocities {
        Vector3<double>(1.0, 0.0, 0.0),
        Vector3<double>(3.0, 0.0, 0.0),
        Vector3<double>(5.0, 0.0, 0.0)
    };
    const std::array<double, 3> temperatures { 0.0, 0.0, 0.0 };

    atlas::copy_host_to_device(positions.data(), particle_probe.pos, positions.size());
    atlas::copy_host_to_device(velocities.data(), particle_probe.vel, velocities.size());
    atlas::copy_host_to_device(temperatures.data(), particle_probe.temperature, temperatures.size());

    searcher.build(particle_probe);

    const auto domain_probe = domain->make_device_probe();
    const auto search_probe = searcher.make_device_probe();

    atlas::system::RmsThermometerOperator<double> {}.measure(domain_probe, search_probe, particle_probe);

    const auto field = atlas::test::copy_device_range(
        domain_probe.field_temperature,
        static_cast<std::size_t>(domain->number_of_cells()));
    const auto particle_temperatures = atlas::test::copy_device_range(
        particle_probe.temperature,
        static_cast<std::size_t>(particle_probe.particle_count));

    const auto first_cell = atlas::SpatialHashingSearcher<double>::linear_key(0, 0, 0, domain->grid_size());
    const auto last_cell  = atlas::SpatialHashingSearcher<double>::linear_key(1, 1, 1, domain->grid_size());

    ASSERT_LT(static_cast<std::size_t>(first_cell), field.size());
    ASSERT_LT(static_cast<std::size_t>(last_cell), field.size());
    EXPECT_NEAR(field[first_cell], 1.0, 1e-12);
    EXPECT_NEAR(field[last_cell], 0.0, 1e-12);
    EXPECT_NEAR(particle_temperatures[0], 1.0, 1e-12);
    EXPECT_NEAR(particle_temperatures[1], 1.0, 1e-12);
    EXPECT_NEAR(particle_temperatures[2], 0.0, 1e-12);
}
