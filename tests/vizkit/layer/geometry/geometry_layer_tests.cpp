#include "../../../utilities/tests_utils.h"

#ifdef ATLAS_ENABLE_VIZKIT

#include <gtest/gtest.h>

TEST(VizkitGeometryLayer, SynchronizeWithoutUnitCopiesLocalPositions) {
    atlas::test::TestVizkitGeometryLayer<float> layer;
    layer.local_positions_public() = {
        atlas::Vector3<float>(1.0f, 2.0f, 3.0f),
        atlas::Vector3<float>(-1.0f, 0.5f, 4.0f)
    };

    std::vector<atlas::Vector3<float>> world_positions;
    const bool changed = layer.synchronize_public(world_positions, 0.0f);

    EXPECT_TRUE(changed);
    ASSERT_EQ(world_positions.size(), 2u);
    EXPECT_TRUE(atlas::test::vec_near(world_positions[0], atlas::Vector3<float>(1.0f, 2.0f, 3.0f), 1e-6f));
    EXPECT_TRUE(atlas::test::vec_near(world_positions[1], atlas::Vector3<float>(-1.0f, 0.5f, 4.0f), 1e-6f));
}

TEST(VizkitGeometryLayer, SynchronizeWithUnitAppliesWorldTransform) {
    atlas::test::TestVizkitGeometryLayer<float> layer(
        atlas::test::make_vizkit_translated_box_unit<float>(atlas::Vector3<float>(3.0f, -2.0f, 1.0f)));
    layer.local_positions_public() = {
        atlas::Vector3<float>(0.0f, 0.0f, 0.0f),
        atlas::Vector3<float>(1.0f, 2.0f, 3.0f)
    };

    std::vector<atlas::Vector3<float>> world_positions;
    const bool changed = layer.synchronize_public(world_positions, 0.0f);

    EXPECT_TRUE(changed);
    ASSERT_EQ(world_positions.size(), 2u);
    EXPECT_TRUE(atlas::test::vec_near(world_positions[0], atlas::Vector3<float>(3.0f, -2.0f, 1.0f), 1e-6f));
    EXPECT_TRUE(atlas::test::vec_near(world_positions[1], atlas::Vector3<float>(4.0f, 0.0f, 4.0f), 1e-6f));
}

#endif
