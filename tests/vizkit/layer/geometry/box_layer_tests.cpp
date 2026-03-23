#include "../../../utilities/tests_utils.h"

#ifdef ATLAS_ENABLE_VIZKIT

#define protected public
#include <vizkit/layer/geometry/box_layer.h>
#undef protected

#include <gtest/gtest.h>

TEST(VizkitBoxLayer, BuildGeometryProducesTwelveLineSegmentsForBox) {
    atlas::vizkit::BoxLayer<float> layer(
        atlas::test::make_vizkit_box_unit<float>(
            atlas::Vector3<float>(-1.0f, -2.0f, -3.0f),
            atlas::Vector3<float>(4.0f, 5.0f, 6.0f)));
    std::vector<atlas::Vector3<float>> positions;

    layer.build_geometry(positions);

    EXPECT_EQ(positions.size(), 24u);
    EXPECT_TRUE(atlas::test::vec_near(positions.front(), atlas::Vector3<float>(-1.0f, -2.0f, -3.0f), 1e-6f));
    EXPECT_TRUE(atlas::test::vec_near(positions.back(), atlas::Vector3<float>(-1.0f, 5.0f, 6.0f), 1e-6f));
}

TEST(VizkitBoxLayer, BuilderRejectsMissingUnit) {
    EXPECT_THROW(
        atlas::vizkit::BoxLayer<float>::builder().build(),
        std::runtime_error);
}

#endif
