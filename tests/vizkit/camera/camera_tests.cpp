#include "../../utilities/tests_utils.h"

#ifdef ATLAS_ENABLE_VIZKIT

#include <vizkit/camera/camera.h>

#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>

TEST(VizkitCamera, BuildMvpProducesFiniteMatrixForValidViewport) {
    atlas::vizkit::Camera camera;
    float mvp[16] = {};

    camera.build_mvp(1280, 720, mvp);

    EXPECT_TRUE(std::all_of(
        std::begin(mvp),
        std::end(mvp),
        [](const float value) { return std::isfinite(value); }));
}

TEST(VizkitCamera, BuildMvpHandlesZeroHeightViewport) {
    atlas::vizkit::Camera camera;
    float mvp[16] = {};

    camera.build_mvp(640, 0, mvp);

    EXPECT_TRUE(std::all_of(
        std::begin(mvp),
        std::end(mvp),
        [](const float value) { return std::isfinite(value); }));
}

#endif
