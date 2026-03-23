#include "../../utilities/tests_utils.h"

#ifdef ATLAS_ENABLE_VIZKIT

#include <vizkit/shader/shader_program.h>

#include <gtest/gtest.h>

#include <type_traits>

TEST(VizkitShaderProgram, TypeIsConstructibleAndDestructible) {
    EXPECT_TRUE((std::is_constructible_v<atlas::vizkit::ShaderProgram, const char*, const char*>));
    EXPECT_TRUE((std::is_destructible_v<atlas::vizkit::ShaderProgram>));
}

#endif
