#pragma once

#include <cstdint>
#include <source_location>
#include <sstream>
#include <stdexcept>
#include <string>

namespace atlas {

enum class LoggingLevel : uint8_t {
    all   = 0,
    debug = 1,
    info  = 2,
    warn  = 3,
    error = 4,
    off   = 5
};

#ifdef ATLAS_ENABLE_LOGGING

class Logger final {
public:
    explicit Logger(LoggingLevel level,
                    std::source_location loc = std::source_location::current()) noexcept
        : _level(level)
        , _loc(loc) { }

    ~Logger();

    Logger(const Logger&) = delete;
    Logger&
    operator=(const Logger&)
        = delete;

    Logger(Logger&&) noexcept = default;

    Logger&
    operator=(Logger&&) noexcept = default;

    template <typename U>
    const Logger&
    operator<<(const U& x) const {
        _buffer << x;
        return *this;
    }

private:
    LoggingLevel _level { LoggingLevel::info };

    std::source_location _loc { std::source_location::current() };

    mutable std::stringstream _buffer {};
};

class Logging {
public:
    static void
    set_info_stream(std::ostream* strm);

    static void
    set_warn_stream(std::ostream* strm);

    static void
    set_error_stream(std::ostream* strm);

    static void
    set_debug_stream(std::ostream* strm);

    static void
    set_all_stream(std::ostream* strm);

    static void
    set_level(LoggingLevel level);

    static void
    mute();

    static void
    unmute();
};

inline Logger
info(std::source_location loc = std::source_location::current()) {
    return Logger(LoggingLevel::info, loc);
}

inline Logger
warn(std::source_location loc = std::source_location::current()) {
    return Logger(LoggingLevel::warn, loc);
}

inline Logger
error(std::source_location loc = std::source_location::current()) {
    return Logger(LoggingLevel::error, loc);
}

inline Logger
debug(std::source_location loc = std::source_location::current()) {
    return Logger(LoggingLevel::debug, loc);
}

bool
should_log(LoggingLevel msg_level) noexcept;

void
emit_log_line(LoggingLevel msg_level,
              const std::source_location& loc,
              const std::string& message);

#else

struct NullLogger {

    template <typename U>
    const NullLogger&
    operator<<(const U&) const {
        return *this;
    }
};

inline constexpr NullLogger atlas_null_logger {};

constexpr const NullLogger&
make_null_logger() noexcept {
    return atlas_null_logger;
}

class Logging {
public:
    static void
    set_info_stream(std::ostream*) { }

    static void
    set_warn_stream(std::ostream*) { }

    static void
    set_error_stream(std::ostream*) { }

    static void
    set_debug_stream(std::ostream*) { }

    static void
    set_all_stream(std::ostream*) { }

    static void
    set_level(LoggingLevel) { }

    static void
    mute() { }

    static void
    unmute() { }
};

inline const NullLogger&
info(std::source_location = std::source_location::current()) {
    return make_null_logger();
}

inline const NullLogger&
warn(std::source_location = std::source_location::current()) {
    return make_null_logger();
}

inline const NullLogger&
error(std::source_location = std::source_location::current()) {
    return make_null_logger();
}

inline const NullLogger&
debug(std::source_location = std::source_location::current()) {
    return make_null_logger();
}

#endif

template <typename ExceptionT>
struct ErrorIfProxy final {

    bool should_throw { false };

    std::source_location loc { std::source_location::current() };

    std::stringstream user_ss {};

    explicit ErrorIfProxy(bool shouldThrow,
                          std::source_location l = std::source_location::current()) noexcept
        : should_throw(shouldThrow)
        , loc(l) { }

    template <typename U>
    ErrorIfProxy&
    operator<<(const U& x) {
        user_ss << x;
        return *this;
    }

    ~ErrorIfProxy() noexcept(false) {
        if (!should_throw) return;

        const std::string user_msg = user_ss.str().empty() ? "ATLAS error" : user_ss.str();

#ifdef ATLAS_ENABLE_LOGGING
        if (::atlas::should_log(::atlas::LoggingLevel::error)) {
            ::atlas::emit_log_line(::atlas::LoggingLevel::error, loc, user_msg);
        }
#endif
        throw ExceptionT(user_msg);
    }
};

struct NoopErrorProxy final {

    template <typename U>
    constexpr const NoopErrorProxy&
    operator<<(const U&) const noexcept {
        return *this;
    }
};

inline constexpr NoopErrorProxy noop_error_proxy {};

template <typename ExceptionT = std::runtime_error>
inline ErrorIfProxy<ExceptionT>
error_if(bool cond, std::source_location loc = std::source_location::current()) noexcept {
    return ErrorIfProxy<ExceptionT>(cond, loc);
}

template <typename ExceptionT = std::runtime_error>
inline ErrorIfProxy<ExceptionT>
check(bool cond, std::source_location loc = std::source_location::current()) noexcept {
    return ErrorIfProxy<ExceptionT>(!cond, loc);
}

#if !defined(NDEBUG)

#define ATLAS_ERROR_IF(cond) \
    ::atlas::error_if((cond), std::source_location::current())
#else

#define ATLAS_ERROR_IF(cond) \
    ((void)sizeof(cond), ::atlas::noop_error_proxy)
#endif

#if !defined(NDEBUG)

#define ATLAS_CHECK(cond) \
    ::atlas::check((cond), std::source_location::current())
#else

#define ATLAS_CHECK(cond) \
    ((void)sizeof(cond), ::atlas::noop_error_proxy)
#endif

}
