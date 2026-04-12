#include "../../utilities/tests_utils.h"

#include <gtest/gtest.h>

#include <atlas/atlas.h>

#include <array>
#include <stdexcept>

TEST(VarianceThermometer, BuilderDefaultsToVarianceType) {
    auto thermometer = atlas::VarianceThermometer<double>::builder().build();

    EXPECT_TRUE(thermometer.is_valid());
    EXPECT_EQ(thermometer.type(), atlas::system::ThermometerType::Variance);
    EXPECT_EQ(thermometer.measure_mode(), atlas::system::MeasureModeType::All);
}

TEST(VarianceThermometer, BuilderMakeHostSharedCreatesValidPointer) {
    const auto thermometer = atlas::VarianceThermometer<double>::builder().make_host_shared();

    ASSERT_NE(thermometer, nullptr);
    EXPECT_TRUE(thermometer->is_valid());
}

TEST(VarianceThermometer, MeasureThrowsForIsothermalDomain) {
    const auto domain = atlas::test::make_isothermal_domain_ptr<double>(300.0);
    auto searcher     = atlas::test::make_single_range_searcher(domain);
    atlas::system::Fluid<double> particle_data(2);

    auto domain_probe   = domain->make_device_probe();
    auto search_probe   = searcher.make_device_probe();
    auto particle_probe = particle_data.make_device_probe();

    auto thermometer = atlas::VarianceThermometer<double>::builder().build();

    EXPECT_THROW(thermometer.measure(domain_probe, search_probe, particle_probe), std::runtime_error);
}

TEST(VarianceThermometer, OperatorComputesCellTemperatureFromVelocityVariance) {
    const auto domain = atlas::Domain<double>::builder()
                            .with_lower_corner(atlas::Vector3<double>(0.0, 0.0, 0.0))
                            .with_upper_corner(atlas::Vector3<double>(1.0, 1.0, 1.0))
                            .with_cell_size(1.0)
                            .make_host_shared();
    auto searcher = atlas::test::make_single_range_searcher(domain);
    atlas::system::Fluid<double> particle_data(2);

    auto& particle_properties = particle_data.particles();
    particle_properties.resize(1);
    particle_properties[0] = atlas::system::MatrialProperties<double>::builder()
                                 .with_mass(2.0)
                                 .build();

    auto particle_probe           = particle_data.make_device_probe();
    particle_probe.particle_count = 2;

    const std::array<atlas::Vector3<double>, 2> positions {
        atlas::Vector3<double>(0.25, 0.25, 0.25),
        atlas::Vector3<double>(0.75, 0.75, 0.75)
    };
    const std::array<atlas::Vector3<double>, 2> velocities {
        atlas::Vector3<double>(3.0, 0.0, 0.0),
        atlas::Vector3<double>(1.0, 0.0, 0.0)
    };
    const std::array<std::size_t, 2> species { 0, 0 };

    atlas::copy_host_to_device(positions.data(), particle_probe.pos, positions.size());
    atlas::copy_host_to_device(velocities.data(), particle_probe.vel, velocities.size());
    atlas::copy_host_to_device(species.data(), particle_probe.species, species.size());

    searcher.build(particle_probe);

    const auto domain_probe = domain->make_device_probe();
    const auto search_probe = searcher.make_device_probe();

    atlas::system::VarianceThermometerOperator<double> {}.measure(domain_probe, search_probe, particle_probe);

    const auto field = atlas::test::copy_device_range(
        domain_probe.field_temperature,
        static_cast<std::size_t>(domain->number_of_cells()));
    const auto particle_temperatures = atlas::test::copy_device_range(
        particle_probe.temperature,
        static_cast<std::size_t>(particle_probe.particle_count));

    const double expected_temperature = 2.0 / (3.0 * static_cast<double>(atlas::boltzmann_constant));
    const auto first_cell = atlas::SpatialHashingSearcher<double>::linear_key(0, 0, 0, domain->grid_size());

    ASSERT_LT(static_cast<std::size_t>(first_cell), field.size());
    EXPECT_NEAR(field[static_cast<std::size_t>(first_cell)], expected_temperature, expected_temperature * 1e-12);
    EXPECT_NEAR(particle_temperatures[0], expected_temperature, expected_temperature * 1e-12);
    EXPECT_NEAR(particle_temperatures[1], expected_temperature, expected_temperature * 1e-12);
}

TEST(VarianceThermometer, FieldModeLeavesParticleTemperaturesUnchanged) {
    const auto domain = atlas::Domain<double>::builder()
                            .with_lower_corner(atlas::Vector3<double>(0.0, 0.0, 0.0))
                            .with_upper_corner(atlas::Vector3<double>(1.0, 1.0, 1.0))
                            .with_cell_size(1.0)
                            .make_host_shared();
    auto searcher = atlas::test::make_single_range_searcher(domain);
    atlas::system::Fluid<double> particle_data(2);

    auto& particle_properties = particle_data.particles();
    particle_properties.resize(1);
    particle_properties[0] = atlas::system::MatrialProperties<double>::builder()
                                 .with_mass(2.0)
                                 .build();

    auto particle_probe           = particle_data.make_device_probe();
    particle_probe.particle_count = 2;

    const std::array<atlas::Vector3<double>, 2> positions {
        atlas::Vector3<double>(0.25, 0.25, 0.25),
        atlas::Vector3<double>(0.75, 0.75, 0.75)
    };
    const std::array<atlas::Vector3<double>, 2> velocities {
        atlas::Vector3<double>(3.0, 0.0, 0.0),
        atlas::Vector3<double>(1.0, 0.0, 0.0)
    };
    const std::array<std::size_t, 2> species { 0, 0 };
    const std::array<double, 2> original_particle_temperatures { 111.0, 222.0 };

    atlas::copy_host_to_device(positions.data(), particle_probe.pos, positions.size());
    atlas::copy_host_to_device(velocities.data(), particle_probe.vel, velocities.size());
    atlas::copy_host_to_device(species.data(), particle_probe.species, species.size());
    atlas::copy_host_to_device(
        original_particle_temperatures.data(),
        particle_probe.temperature,
        original_particle_temperatures.size());

    searcher.build(particle_probe);

    const auto domain_probe = domain->make_device_probe();
    const auto search_probe = searcher.make_device_probe();

    atlas::system::VarianceThermometerOperator<double> {}.measure(
        domain_probe,
        search_probe,
        particle_probe,
        atlas::system::MeasureModeType::Field);

    const auto particle_temperatures = atlas::test::copy_device_range(
        particle_probe.temperature,
        static_cast<std::size_t>(particle_probe.particle_count));

    EXPECT_DOUBLE_EQ(particle_temperatures[0], original_particle_temperatures[0]);
    EXPECT_DOUBLE_EQ(particle_temperatures[1], original_particle_temperatures[1]);
}
