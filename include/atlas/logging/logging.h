#pragma once

#include <cstdint>
#include <source_location>
#include <sstream>
#include <stdexcept>
#include <string>

namespace atlas {

/**
 * @file logging.h
 * @brief Lightweight logging front-end and header-only error helpers.
 *
 * @details
 * This header provides:
 * - A small, RAII-style `Logger` interface (enabled only when `ATLAS_ENABLE_LOGGING` is defined).
 * - A compile-time zero-cost fallback (`NullLogger`) when logging is disabled.
 * - Header-only `error_if()` / `check()` helpers that throw exceptions with a streamed message.
 * - Debug-only convenience macros (`ATLAS_ERROR_IF`, `ATLAS_CHECK`) that compile out in Release
 *   (following the standard `NDEBUG` convention).
 *
 * The logging backend (formatting, sinks, synchronization) is intentionally kept out-of-line
 * in a `.cpp` translation unit to keep this header light and widely includable.
 */

/**
 * @brief Logging severity levels.
 *
 * @details
 * Ordering is significant: lower values are more verbose.
 *
 * Typical meaning:
 * - `All`   : log everything
 * - `Debug` : debug diagnostics
 * - `Info`  : general information
 * - `Warn`  : recoverable issues
 * - `Error` : serious failures
 * - `Off`   : disable all logging
 */
enum class LoggingLevel : uint8_t {
    All   = 0,
    Debug = 1,
    Info  = 2,
    Warn  = 3,
    Error = 4,
    Off   = 5
};

#ifdef ATLAS_ENABLE_LOGGING

// =====================================================
// Logger (light header; heavy implementation in .cpp)
// =====================================================

/**
 * @brief RAII-style logger object.
 *
 * @details
 * A `Logger` instance acts like a temporary stream:
 * - You construct it (typically via `atlas::logger::info()` / `warn()` / `error()` / `debug()`).
 * - You stream data into it with `operator<<`.
 * - When the temporary is destroyed, it emits exactly one formatted log line.
 *
 * This pattern enables concise call-site usage without manual flushing:
 * @code
 * atlas::logger::info() << "value = " << x;
 * atlas::logger::warn() << "clamped: " << a << " -> " << b;
 * @endcode
 *
 * @note
 * Formatting, timestamps, thread id, coloring, sink routing, etc. should be handled by the
 * out-of-line backend implementation (see `atlas::detail::emit_log_line`).
 */
class Logger final {
public:
    /**
     * @brief Construct a logger with a given severity level.
     *
     * @param level Logging severity for this message.
     * @param loc   Source location captured at the call site (defaults to current).
     *
     * @note
     * The captured location is used by the backend to annotate the output
     * (e.g., file name, line number, function name).
     */
    explicit Logger(LoggingLevel level,
                    std::source_location loc = std::source_location::current()) noexcept
        : _level(level)
        , _loc(loc) { }

    /**
     * @brief Destructor emits the accumulated log line.
     *
     * @details
     * Defined out-of-line to keep this header lightweight. The destructor typically:
     * - Checks the global logging threshold / mute state.
     * - Formats a single line including location and severity.
     * - Writes it to the configured sink.
     *
     * @warning
     * The destructor must not throw.
     */
    ~Logger();

    Logger(const Logger&) = delete;
    Logger&
    operator=(const Logger&)
        = delete;

    Logger(Logger&&) noexcept = default;
    Logger&
    operator=(Logger&&) noexcept = default;

    /**
     * @brief Stream arbitrary values into the log message.
     *
     * @tparam U Any type supported by `std::stringstream::operator<<`.
     * @param x  Value to append to this message.
     * @return Const reference to `*this` so calls can be chained.
     *
     * @note
     * This operator is `const` so the logger can be used as a temporary.
     * The internal buffer is therefore `mutable`.
     */
    template <typename U>
    const Logger&
    operator<<(const U& x) const {
        _buffer << x;
        return *this;
    }

private:
    /// Selected logging level for this message.
    LoggingLevel _level { LoggingLevel::Info };

    /// Source location captured at construction time.
    std::source_location _loc { std::source_location::current() };

    /// Accumulated user message.
    mutable std::stringstream _buffer {};
};

/**
 * @brief Global logging configuration utilities.
 *
 * @details
 * This type exposes global configuration points for the logging backend,
 * such as per-level sinks and the global severity threshold.
 *
 * Thread-safety requirements depend on the backend implementation; typically:
 * - The backend should synchronize access to sink pointers and level state.
 * - Calls to these functions should be safe at runtime, but may be restricted
 *   to initialization time depending on your engine policy.
 */
class Logging {
public:
    /**
     * @brief Set output stream for Info messages.
     * @param strm Stream to use; `nullptr` restores the default stream.
     */
    static void
    set_info_stream(std::ostream* strm);

    /**
     * @brief Set output stream for Warn messages.
     * @param strm Stream to use; `nullptr` restores the default stream.
     */
    static void
    set_warn_stream(std::ostream* strm);

    /**
     * @brief Set output stream for Error messages.
     * @param strm Stream to use; `nullptr` restores the default stream.
     */
    static void
    set_error_stream(std::ostream* strm);

    /**
     * @brief Set output stream for Debug messages.
     * @param strm Stream to use; `nullptr` restores the default stream.
     */
    static void
    set_debug_stream(std::ostream* strm);

    /**
     * @brief Assign the same stream to all log levels.
     * @param strm Stream to use; `nullptr` restores defaults.
     */
    static void
    set_all_stream(std::ostream* strm);

    /**
     * @brief Set the global logging level threshold.
     *
     * @details
     * Messages more verbose than the threshold are suppressed by the backend.
     *
     * @param level New threshold.
     */
    static void
    set_level(LoggingLevel level);

    /**
     * @brief Disable all logging output (equivalent to setting level to `Off`).
     */
    static void
    mute();

    /**
     * @brief Enable logging for all levels (equivalent to setting level to `All`).
     */
    static void
    unmute();
};

/**
 * @brief Convenience logging entry points.
 *
 * @details
 * These helpers return a `Logger` temporary. Users can stream values into it
 * and the message is emitted on destruction.
 *
 * Placing these in `atlas::logger` avoids clashes with functions such as
 * `std::log` or math libraries.
 */
namespace logger {

    /**
     * @brief Create an Info-level logger.
     * @param loc Source location (defaults to call site).
     */
    inline Logger
    info(std::source_location loc = std::source_location::current()) {
        return Logger(LoggingLevel::Info, loc);
    }

    /**
     * @brief Create a Warn-level logger.
     * @param loc Source location (defaults to call site).
     */
    inline Logger
    warn(std::source_location loc = std::source_location::current()) {
        return Logger(LoggingLevel::Warn, loc);
    }

    /**
     * @brief Create an Error-level logger.
     * @param loc Source location (defaults to call site).
     */
    inline Logger
    error(std::source_location loc = std::source_location::current()) {
        return Logger(LoggingLevel::Error, loc);
    }

    /**
     * @brief Create a Debug-level logger.
     * @param loc Source location (defaults to call site).
     */
    inline Logger
    debug(std::source_location loc = std::source_location::current()) {
        return Logger(LoggingLevel::Debug, loc);
    }

} // namespace logger

namespace detail {

    /**
     * @brief Check whether a message at the given level should be logged.
     *
     * @details
     * Implemented in the backend translation unit. It typically compares
     * `msg_level` to the current global threshold and respects mute state.
     *
     * @param msg_level Severity level of the candidate message.
     * @return `true` if the message should be emitted; otherwise `false`.
     */
    bool
    should_log(LoggingLevel msg_level) noexcept;

    /**
     * @brief Emit a fully formatted log line.
     *
     * @details
     * Implemented in the backend translation unit. This function is responsible for:
     * - Formatting the message (severity label, timestamp, etc.).
     * - Attaching source location information (file/line/function).
     * - Routing to the correct sink/stream for the message level.
     *
     * @param msg_level Message severity.
     * @param loc       Source location.
     * @param message   User-provided message content.
     */
    void
    emit_log_line(LoggingLevel msg_level,
                  const std::source_location& loc,
                  const std::string& message);

} // namespace detail

#else // =====================================================
// Logging disabled
// =====================================================

/**
 * @brief Null logger used when logging is disabled at compile time.
 *
 * @details
 * This type mimics the streaming interface of `Logger` but discards everything.
 * Because the implementation is trivial and `constexpr`, modern compilers will
 * optimize away chained `operator<<` calls entirely.
 */
struct NullLogger {
    /**
     * @brief Ignore an appended value and return `*this` for chaining.
     */
    template <typename U>
    const NullLogger&
    operator<<(const U&) const {
        return *this;
    }
};

/**
 * @brief Shared constexpr null logger instance.
 */
inline constexpr NullLogger atlas_null_logger {};

/**
 * @brief Return a reference to the null logger.
 *
 * @return Const reference to a shared null logger instance.
 */
constexpr const NullLogger&
make_null_logger() noexcept {
    return atlas_null_logger;
}

/**
 * @brief Stub logging configuration when logging is disabled.
 *
 * @details
 * All functions are no-ops to preserve API compatibility.
 */
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

namespace logger {

    /**
     * @brief Null Info logger.
     */
    inline const NullLogger&
    info(std::source_location = std::source_location::current()) {
        return make_null_logger();
    }

    /**
     * @brief Null Warn logger.
     */
    inline const NullLogger&
    warn(std::source_location = std::source_location::current()) {
        return make_null_logger();
    }

    /**
     * @brief Null Error logger.
     */
    inline const NullLogger&
    error(std::source_location = std::source_location::current()) {
        return make_null_logger();
    }

    /**
     * @brief Null Debug logger.
     */
    inline const NullLogger&
    debug(std::source_location = std::source_location::current()) {
        return make_null_logger();
    }

} // namespace logger

#endif // ATLAS_ENABLE_LOGGING

// =====================================================
// error_if / check (template, header-only)
// =====================================================
namespace detail {

    /**
     * @brief Proxy object implementing deferred error handling.
     *
     * @details
     * `ErrorIfProxy` is returned by `error_if()` / `check()` and supports:
     * @code
     * atlas::check(x > 0) << "x must be positive, got " << x;
     * @endcode
     *
     * The proxy accumulates the message using a `std::stringstream`. When the proxy
     * is destroyed, it throws an exception if `should_throw == true`.
     *
     * If logging is enabled, an Error-level log line is also emitted before throwing
     * (subject to the global threshold).
     *
     * @tparam ExceptionT Exception type to throw (e.g., `std::runtime_error`).
     *
     * @warning
     * Because the exception is thrown from the destructor, this is intended to be used
     * as a temporary at statement scope. Avoid keeping it in a named variable unless you
     * clearly control its lifetime.
     */
    template <typename ExceptionT>
    struct ErrorIfProxy final {
        /// Whether an exception should be thrown on destruction.
        bool should_throw { false };

        /// Source location of the check.
        std::source_location loc { std::source_location::current() };

        /// User-provided message stream.
        std::stringstream user_ss {};

        /**
         * @brief Construct an ErrorIfProxy.
         *
         * @param shouldThrow Whether to throw on destruction.
         * @param l           Source location (defaults to call site).
         */
        explicit ErrorIfProxy(bool shouldThrow,
                              std::source_location l = std::source_location::current()) noexcept
            : should_throw(shouldThrow)
            , loc(l) { }

        /**
         * @brief Append to the error message.
         *
         * @tparam U Any streamable type.
         * @param x  Value to append.
         * @return Reference to this proxy for chaining.
         */
        template <typename U>
        ErrorIfProxy&
        operator<<(const U& x) {
            user_ss << x;
            return *this;
        }

        /**
         * @brief Destructor throws if the condition was met.
         *
         * @details
         * - If no custom message is provided, uses a default `"ATLAS error"`.
         * - When logging is enabled and the global policy allows it, emits an Error log line.
         * - Throws `ExceptionT` with the user message.
         *
         * @throws ExceptionT if `should_throw == true`.
         */
        ~ErrorIfProxy() noexcept(false) {
            if (!should_throw) return;

            const std::string user_msg = user_ss.str().empty() ? "ATLAS error" : user_ss.str();

#ifdef ATLAS_ENABLE_LOGGING
            if (::atlas::detail::should_log(::atlas::LoggingLevel::Error)) {
                ::atlas::detail::emit_log_line(::atlas::LoggingLevel::Error, loc, user_msg);
            }
#endif
            throw ExceptionT(user_msg);
        }
    };

    // -----------------------------------------------------
    // Debug-only check helpers (macro-friendly)
    // -----------------------------------------------------

    /**
     * @brief No-op proxy used when checks are compiled out.
     *
     * @details
     * Supports `operator<<` chaining and is intended for use by the debug-only macros.
     * The macro definitions ensure the condition expression is not evaluated in Release.
     */
    struct NoopErrorProxy final {
        /**
         * @brief Ignore an appended value and return `*this` for chaining.
         */
        template <typename U>
        constexpr const NoopErrorProxy&
        operator<<(const U&) const noexcept {
            return *this;
        }
    };

    /// Shared constexpr no-op proxy instance.
    inline constexpr NoopErrorProxy noop_error_proxy {};

} // namespace detail

/**
 * @brief Throw an exception if a condition is true.
 *
 * @details
 * Returns a proxy object that can be streamed into to build a message. The proxy throws
 * from its destructor if `cond` is true.
 *
 * Example:
 * @code
 * atlas::error_if(ptr == nullptr) << "ptr must not be null";
 * @endcode
 *
 * @tparam ExceptionT Exception type (default: `std::runtime_error`).
 * @param cond Condition triggering the error.
 * @param loc  Source location (defaults to call site).
 * @return Proxy that throws on destruction when `cond` is true.
 */
template <typename ExceptionT = std::runtime_error>
inline detail::ErrorIfProxy<ExceptionT>
error_if(bool cond, std::source_location loc = std::source_location::current()) noexcept {
    return detail::ErrorIfProxy<ExceptionT>(cond, loc);
}

/**
 * @brief Assert-like check that throws if the condition is false.
 *
 * @details
 * This is the logical inverse of `error_if()`. It returns a proxy that throws when `cond` is false.
 *
 * Example:
 * @code
 * atlas::check(n > 0) << "n must be positive, got " << n;
 * @endcode
 *
 * @tparam ExceptionT Exception type (default: `std::runtime_error`).
 * @param cond Condition expected to be true.
 * @param loc  Source location (defaults to call site).
 * @return Proxy that throws on destruction when `cond` is false.
 */
template <typename ExceptionT = std::runtime_error>
inline detail::ErrorIfProxy<ExceptionT>
check(bool cond, std::source_location loc = std::source_location::current()) noexcept {
    return detail::ErrorIfProxy<ExceptionT>(!cond, loc);
}

// =====================================================
// Debug-only macros (compile-out in Release)
// =====================================================

/**
 * @def ATLAS_ERROR_IF(cond)
 * @brief Debug-only `error_if()` convenience macro.
 *
 * @details
 * - In Debug (when `NDEBUG` is not defined): expands to `atlas::error_if(cond, current_location)`.
 * - In Release: does not evaluate `cond` and returns a no-op proxy that supports `<<`.
 *
 * Example:
 * @code
 * ATLAS_ERROR_IF(x < 0) << "x must be >= 0, got " << x;
 * @endcode
 */
#if !defined(NDEBUG)
#define ATLAS_ERROR_IF(cond) \
    ::atlas::error_if((cond), std::source_location::current())
#else
#define ATLAS_ERROR_IF(cond) \
    ((void)sizeof(cond), ::atlas::detail::noop_error_proxy)
#endif

/**
 * @def ATLAS_CHECK(cond)
 * @brief Debug-only `check()` convenience macro.
 *
 * @details
 * - In Debug (when `NDEBUG` is not defined): expands to `atlas::check(cond, current_location)`.
 * - In Release: does not evaluate `cond` and returns a no-op proxy that supports `<<`.
 *
 * Example:
 * @code
 * ATLAS_CHECK(ptr != nullptr) << "ptr must not be null";
 * @endcode
 */
#if !defined(NDEBUG)
#define ATLAS_CHECK(cond) \
    ::atlas::check((cond), std::source_location::current())
#else
#define ATLAS_CHECK(cond) \
    ((void)sizeof(cond), ::atlas::detail::noop_error_proxy)
#endif

} // namespace atlas
