#include "../utilities/tests_utils.h"

#include <atlas/atlas.h>
#include <gtest/gtest.h>

using namespace atlas;

TEST(Domain, DeviceProbeLeavesFieldForceNullWhenUnset) {
    auto domain = atlas::Domain<double>::builder()
                      .with_lower_corner(Vector3<double>(0.0, 0.0, 0.0))
                      .with_upper_corner(Vector3<double>(1.0, 1.0, 1.0))
                      .with_cell_size(1.0)
                      .build();

    auto probe = domain.make_device_probe();

    EXPECT_EQ(probe.field_force, nullptr);
    EXPECT_TRUE(domain.field_force().empty());
}

TEST(Domain, FieldForceSetterAndGetterStoreConfiguredBuffer) {
    auto domain = atlas::Domain<double>::builder()
                      .with_lower_corner(Vector3<double>(0.0, 0.0, 0.0))
                      .with_upper_corner(Vector3<double>(1.0, 1.0, 1.0))
                      .with_cell_size(1.0)
                      .build();

    const HostBuffer<Vector3<double>> field_force {
        Vector3<double>(1.0, 0.0, 0.0),
        Vector3<double>(0.0, 2.0, 0.0),
        Vector3<double>(0.0, 0.0, 3.0),
        Vector3<double>(4.0, 5.0, 6.0),
        Vector3<double>(7.0, 8.0, 9.0),
        Vector3<double>(10.0, 11.0, 12.0),
        Vector3<double>(13.0, 14.0, 15.0),
        Vector3<double>(16.0, 17.0, 18.0)
    };

    domain.set_field_force(field_force);

    ASSERT_EQ(domain.field_force().size(), field_force.size());
    EXPECT_TRUE(test::point_buffers_near(test::copy_device_buffer(domain.field_force()), field_force, 1e-12));

    auto probe = domain.make_device_probe();
    ASSERT_NE(probe.field_force, nullptr);

    const auto probe_field_force =
        test::copy_device_range(probe.field_force, static_cast<std::size_t>(domain.number_of_cells()));
    EXPECT_TRUE(test::point_buffers_near(probe_field_force, field_force, 1e-12));
}
