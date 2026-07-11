#include <atlas/logging/logging.h>

#include <gtest/gtest.h>

#include <cstdint>
#include <sstream>
#include <stdexcept>
#include <string>
#include <type_traits>

namespace {

using atlas::LoggingLevel;

// Underlying value of a LoggingLevel, used to assert the severity ordering the
// header documents as load-bearing for the threshold comparison.
constexpr std::uint8_t
raw(LoggingLevel level) {
    return static_cast<std::uint8_t>(level);
}

}

// The fixed uint8_t underlying type is relied upon by the source, which compares
// levels through a static_cast to it.
static_assert(std::is_same_v<std::underlying_type_t<LoggingLevel>, std::uint8_t>,
              "LoggingLevel must have a uint8_t underlying type");

TEST(LoggingLevel, OrderedByIncreasingSeverity) {
    EXPECT_LT(raw(LoggingLevel::all), raw(LoggingLevel::debug));
    EXPECT_LT(raw(LoggingLevel::debug), raw(LoggingLevel::info));
    EXPECT_LT(raw(LoggingLevel::info), raw(LoggingLevel::warn));
    EXPECT_LT(raw(LoggingLevel::warn), raw(LoggingLevel::error));
    EXPECT_LT(raw(LoggingLevel::error), raw(LoggingLevel::off));
}

// --- check / error_if guards (available in every build; the throw is not gated
// on ATLAS_ENABLE_LOGGING, only the log emission is) --------------------------

TEST(Check, PassingConditionDoesNotThrow) {
    EXPECT_NO_THROW(atlas::check(true));
}

TEST(Check, FailingConditionThrowsRuntimeError) {
    EXPECT_THROW(atlas::check(false), std::runtime_error);
}

TEST(Check, UsesCustomExceptionType) {
    EXPECT_THROW(atlas::check<std::invalid_argument>(false), std::invalid_argument);
}

TEST(Check, StreamsMessageIntoException) {
    try {
        atlas::check<std::runtime_error>(false) << "bad value " << 42;
        FAIL() << "check(false) should have thrown";
    } catch (const std::runtime_error& e) {
        EXPECT_STREQ(e.what(), "bad value 42");
    }
}

TEST(Check, UsesDefaultMessageWhenNoneStreamed) {
    try {
        atlas::check(false);
        FAIL() << "check(false) should have thrown";
    } catch (const std::runtime_error& e) {
        EXPECT_STREQ(e.what(), "ATLAS error");
    }
}

TEST(ErrorIf, TrueConditionThrows) {
    EXPECT_THROW(atlas::error_if(true), std::runtime_error);
}

TEST(ErrorIf, FalseConditionDoesNotThrow) {
    EXPECT_NO_THROW(atlas::error_if(false));
}

// --- Runtime configuration facade and emission -------------------------------
// These exercise the active logging machinery, which only exists when
// ATLAS_ENABLE_LOGGING is defined. Every test build enables it (the debug
// presets set ATLAS_LOGGING=ON, which exports ATLAS_ENABLE_LOGGING from
// atlas::core); the guard keeps this translation unit compilable if it is not.
#ifdef ATLAS_ENABLE_LOGGING

namespace {

// Restores the process-wide logging configuration after each test, since it is
// global mutable state shared across every case in the executable.
class LoggingConfig : public ::testing::Test {
protected:
    void
    SetUp() override {
        atlas::Logging::unmute();
        atlas::Logging::set_all_stream(nullptr);
    }

    void
    TearDown() override {
        atlas::Logging::set_all_stream(nullptr);
        atlas::Logging::unmute();
    }
};

}

TEST_F(LoggingConfig, ShouldLogRespectsThreshold) {
    atlas::Logging::set_level(LoggingLevel::all);
    EXPECT_TRUE(atlas::should_log(LoggingLevel::debug));
    EXPECT_TRUE(atlas::should_log(LoggingLevel::warn));

    atlas::Logging::set_level(LoggingLevel::error);
    EXPECT_FALSE(atlas::should_log(LoggingLevel::warn));
    EXPECT_TRUE(atlas::should_log(LoggingLevel::error));

    atlas::Logging::mute();
    EXPECT_FALSE(atlas::should_log(LoggingLevel::error));
}

TEST_F(LoggingConfig, WarnFactoryEmitsToConfiguredStream) {
    std::ostringstream sink;
    atlas::Logging::set_all_stream(&sink);

    atlas::warn() << "hello world";

    const std::string line = sink.str();
    EXPECT_NE(line.find("hello world"), std::string::npos);
    EXPECT_NE(line.find("WARN"), std::string::npos);
}

TEST_F(LoggingConfig, MutedFactorySuppressesEmission) {
    std::ostringstream sink;
    atlas::Logging::set_all_stream(&sink);
    atlas::Logging::mute();

    atlas::warn() << "should not appear";

    EXPECT_TRUE(sink.str().empty());
}

TEST_F(LoggingConfig, EmptyLineIsNotEmitted) {
    std::ostringstream sink;
    atlas::Logging::set_all_stream(&sink);

    atlas::warn();

    EXPECT_TRUE(sink.str().empty());
}

#endif
