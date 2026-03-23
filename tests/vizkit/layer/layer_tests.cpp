#include "../../utilities/tests_utils.h"

#ifdef ATLAS_ENABLE_VIZKIT

#include <gtest/gtest.h>

TEST(VizkitLayer, DerivedLayerReceivesInitAndUpdateCalls) {
    atlas::test::DummyVizkitLayer<float> layer;
    atlas::vizkit::Camera camera;

    layer.init(nullptr, camera);
    layer.update(nullptr, camera, 0.25f);
    layer.shutdown();

    EXPECT_TRUE(layer.init_called);
    EXPECT_TRUE(layer.update_called);
    EXPECT_EQ(layer.last_window, nullptr);
    EXPECT_FLOAT_EQ(layer.last_dt, 0.25f);
}

#endif
