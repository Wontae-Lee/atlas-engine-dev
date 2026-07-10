#include <atlas/material/material_type.h>

#include <gtest/gtest.h>

#include <type_traits>

namespace {

using atlas::MaterialType;

}

static_assert(std::is_same_v<std::underlying_type_t<MaterialType>, int>,
              "MaterialType must be backed by int for host/device stability");

TEST(MaterialType, EnumeratorsAreOrderedFromZero) {
    // The tag value doubles as the DeviceVariant active-member index and is the
    // serialized discriminant, so this order is a stability contract.
    EXPECT_EQ(static_cast<int>(MaterialType::molecule), 0);
    EXPECT_EQ(static_cast<int>(MaterialType::atom), 1);
    EXPECT_EQ(static_cast<int>(MaterialType::ion), 2);
    EXPECT_EQ(static_cast<int>(MaterialType::neutron), 3);
    EXPECT_EQ(static_cast<int>(MaterialType::solid), 4);
}
