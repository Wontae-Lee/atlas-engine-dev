#pragma once

#include <cstdint>
#include <source_location>
#include <sstream>
#include <stdexcept>
#include <string>

namespace atlas {

/**
 * @brief Severity levels used by the Atlas logging system.
 *
 * @details
 * The ordering is intentional and supports threshold-style filtering:
 * lower numeric values represent more permissive logging configurations.
 *
 * Typical interpretation:
 * - @ref All   : allow every message
 * - @ref Debug : allow debug and higher-severity messages
 * - @ref Info  : allow info and higher-severity messages
 * - @ref Warn  : allow warning and error messages
 * - @ref Error : allow only error messages
 * - @ref Off   : disable all logging output
 */
enum class LoggingLevel : uint8_t {
    All   = 0, ///< Emit all log messages.
    Debug = 1, ///< Emit debug, info, warning, and error messages.
    Info  = 2, ///< Emit info, warning, and error messages.
    Warn  = 3, ///< Emit warning and error messages.
    Error = 4, ///< Emit only error messages.
    Off   = 5  ///< Emit no messages.
};

#ifdef ATLAS_ENABLE_LOGGING

/**
 * @brief RAII-style transient logger that buffers a message and emits it on destruction.
 *
 * @details
 * `Logger` is designed for stream-style usage:
 * @code
 * atlas::logger::info() << "step=" << step << ", value=" << value;
 * @endcode
 *
 * The object accumulates streamed text into an internal buffer and flushes the
 * completed line in its destructor. This allows convenient single-expression
 * logging with automatic source-location capture.
 *
 * Instances are moveable but not copyable.
 */
class Logger final {
public:
    /**
     * @brief Construct a logger for a specific severity level.
     *
     * @param level Log severity associated with the message.
     * @param loc Source location captured at the call site.
     */
    explicit Logger(LoggingLevel level,
                    std::source_location loc = std::source_location::current()) noexcept
        : _level(level)
        , _loc(loc) { }

    /**
     * @brief Flush the buffered message when the logger goes out of scope.
     *
     * @details
     * The implementation is expected to:
     * - inspect the current logging configuration,
     * - format the message with source-location metadata,
     * - emit the message to the configured stream(s) if enabled.
     */
    ~Logger();

    Logger(const Logger&) = delete; ///< Copy construction is disabled.
    Logger&
    operator=(const Logger&)
        = delete; ///< Copy assignment is disabled.

    /**
     * @brief Move constructor.
     */
    Logger(Logger&&) noexcept = default;

    /**
     * @brief Move assignment operator.
     */
    Logger&
    operator=(Logger&&) noexcept = default;

    /**
     * @brief Append a value to the buffered log message.
     *
     * @tparam U Streamable value type.
     * @param x Value to append.
     * @return Const reference to this logger to support chained streaming.
     *
     * @details
     * The buffer is mutable so streaming remains possible even though temporary
     * logger objects are commonly treated as const in chained expressions.
     */
    template <typename U>
    const Logger&
    operator<<(const U& x) const {
        _buffer << x;
        return *this;
    }

private:
    LoggingLevel _level { LoggingLevel::Info };
    ///< Severity associated with this buffered message.

    std::source_location _loc { std::source_location::current() };
    ///< Source location captured at logger creation time.

    mutable std::stringstream _buffer {};
    ///< Internal stream buffer holding the message text until destruction.
};

/**
 * @brief Global configuration interface for the logging subsystem.
 *
 * @details
 * This class provides process-wide controls for:
 * - output stream routing by severity,
 * - active severity threshold,
 * - temporary muting/unmuting.
 *
 * All functions are static because the logging system is used as a singleton-style
 * facility rather than through explicit object instances.
 */
class Logging {
public:
    /**
     * @brief Set the output stream used for info-level messages.
     *
     * @param strm Destination stream. May be implementation-defined if null.
     */
    static void
    set_info_stream(std::ostream* strm);

    /**
     * @brief Set the output stream used for warning-level messages.
     *
     * @param strm Destination stream. May be implementation-defined if null.
     */
    static void
    set_warn_stream(std::ostream* strm);

    /**
     * @brief Set the output stream used for error-level messages.
     *
     * @param strm Destination stream. May be implementation-defined if null.
     */
    static void
    set_error_stream(std::ostream* strm);

    /**
     * @brief Set the output stream used for debug-level messages.
     *
     * @param strm Destination stream. May be implementation-defined if null.
     */
    static void
    set_debug_stream(std::ostream* strm);

    /**
     * @brief Route all severity levels to the same output stream.
     *
     * @param strm Destination stream. May be implementation-defined if null.
     */
    static void
    set_all_stream(std::ostream* strm);

    /**
     * @brief Set the active logging threshold.
     *
     * @param level New global logging level.
     */
    static void
    set_level(LoggingLevel level);

    /**
     * @brief Temporarily disable logging output.
     */
    static void
    mute();

    /**
     * @brief Re-enable logging output after a prior call to @ref mute.
     */
    static void
    unmute();
};

namespace logger {

    /**
     * @brief Create an info-level logger.
     *
     * @param loc Source location captured at the call site.
     * @return Temporary logger configured for @ref LoggingLevel::Info.
     */
    inline Logger
    info(std::source_location loc = std::source_location::current()) {
        return Logger(LoggingLevel::Info, loc);
    }

    /**
     * @brief Create a warning-level logger.
     *
     * @param loc Source location captured at the call site.
     * @return Temporary logger configured for @ref LoggingLevel::Warn.
     */
    inline Logger
    warn(std::source_location loc = std::source_location::current()) {
        return Logger(LoggingLevel::Warn, loc);
    }

    /**
     * @brief Create an error-level logger.
     *
     * @param loc Source location captured at the call site.
     * @return Temporary logger configured for @ref LoggingLevel::Error.
     */
    inline Logger
    error(std::source_location loc = std::source_location::current()) {
        return Logger(LoggingLevel::Error, loc);
    }

    /**
     * @brief Create a debug-level logger.
     *
     * @param loc Source location captured at the call site.
     * @return Temporary logger configured for @ref LoggingLevel::Debug.
     */
    inline Logger
    debug(std::source_location loc = std::source_location::current()) {
        return Logger(LoggingLevel::Debug, loc);
    }

} // namespace logger

namespace detail {

    /**
     * @brief Check whether a message of a given severity should currently be emitted.
     *
     * @param msg_level Severity of the message being considered.
     * @return `true` if the message should be logged, otherwise `false`.
     */
    bool
    should_log(LoggingLevel msg_level) noexcept;

    /**
     * @brief Emit one formatted log line.
     *
     * @param msg_level Severity of the message.
     * @param loc Source location associated with the message.
     * @param message Fully formatted user message text.
     */
    void
    emit_log_line(LoggingLevel msg_level,
                  const std::source_location& loc,
                  const std::string& message);

} // namespace detail

#else

/**
 * @brief No-op logger used when logging is compiled out.
 *
 * @details
 * This type absorbs all streamed values and performs no work. It preserves the
 * syntax of stream-style logging calls without generating output or side effects.
 */
struct NullLogger {

    /**
     * @brief Ignore streamed input and return the same no-op logger.
     *
     * @tparam U Arbitrary stream-like input type.
     * @param U unused streamed value.
     * @return Reference to this no-op logger.
     */
    template <typename U>
    const NullLogger&
    operator<<(const U&) const {
        return *this;
    }
};

/**
 * @brief Singleton no-op logger instance.
 */
inline constexpr NullLogger atlas_null_logger {};

/**
 * @brief Return a reference to the shared no-op logger.
 *
 * @return Reference to @ref atlas_null_logger.
 */
constexpr const NullLogger&
make_null_logger() noexcept {
    return atlas_null_logger;
}

/**
 * @brief Stub logging configuration API used when logging is disabled.
 *
 * @details
 * All functions are intentionally empty so code can still call the logging
 * configuration interface without conditional compilation at call sites.
 */
class Logging {
public:
    /**
     * @brief No-op stub for setting the info stream.
     *
     * @param Unused stream pointer.
     */
    static void
    set_info_stream(std::ostream*) { }

    /**
     * @brief No-op stub for setting the warning stream.
     *
     * @param Unused stream pointer.
     */
    static void
    set_warn_stream(std::ostream*) { }

    /**
     * @brief No-op stub for setting the error stream.
     *
     * @param Unused stream pointer.
     */
    static void
    set_error_stream(std::ostream*) { }

    /**
     * @brief No-op stub for setting the debug stream.
     *
     * @param Unused stream pointer.
     */
    static void
    set_debug_stream(std::ostream*) { }

    /**
     * @brief No-op stub for setting all streams at once.
     *
     * @param Unused stream pointer.
     */
    static void
    set_all_stream(std::ostream*) { }

    /**
     * @brief No-op stub for changing the logging level.
     *
     * @param Unused level value.
     */
    static void
    set_level(LoggingLevel) { }

    /**
     * @brief No-op stub for muting logging.
     */
    static void
    mute() { }

    /**
     * @brief No-op stub for unmuting logging.
     */
    static void
    unmute() { }
};

namespace logger {

    /**
     * @brief Return a no-op info logger when logging is disabled.
     *
     * @param Unused source location.
     * @return Reference to the shared null logger.
     */
    inline const NullLogger&
    info(std::source_location = std::source_location::current()) {
        return make_null_logger();
    }

    /**
     * @brief Return a no-op warning logger when logging is disabled.
     *
     * @param Unused source location.
     * @return Reference to the shared null logger.
     */
    inline const NullLogger&
    warn(std::source_location = std::source_location::current()) {
        return make_null_logger();
    }

    /**
     * @brief Return a no-op error logger when logging is disabled.
     *
     * @param Unused source location.
     * @return Reference to the shared null logger.
     */
    inline const NullLogger&
    error(std::source_location = std::source_location::current()) {
        return make_null_logger();
    }

    /**
     * @brief Return a no-op debug logger when logging is disabled.
     *
     * @param Unused source location.
     * @return Reference to the shared null logger.
     */
    inline const NullLogger&
    debug(std::source_location = std::source_location::current()) {
        return make_null_logger();
    }

} // namespace logger

#endif

namespace detail {

    /**
     * @brief Proxy object that throws an exception from its destructor when triggered.
     *
     * @tparam ExceptionT Exception type to throw.
     *
     * @details
     * This proxy enables stream-style assertion/error construction such as:
     * @code
     * atlas::check(cond) << "expected positive value, got " << x;
     * @endcode
     *
     * If `should_throw` is `true` at destruction time:
     * - the accumulated message is finalized,
     * - an error log may be emitted,
     * - `ExceptionT` is thrown with that message.
     *
     * If `should_throw` is `false`, destruction is a no-op.
     */
    template <typename ExceptionT>
    struct ErrorIfProxy final {

        bool should_throw { false };
        ///< Whether destruction should raise an exception.

        std::source_location loc { std::source_location::current() };
        ///< Source location associated with the check/error site.

        std::stringstream user_ss {};
        ///< User-provided message buffer built via stream insertion.

        /**
         * @brief Construct the proxy.
         *
         * @param shouldThrow Whether the destructor should throw.
         * @param l Source location associated with the proxy.
         */
        explicit ErrorIfProxy(bool shouldThrow,
                              std::source_location l = std::source_location::current()) noexcept
            : should_throw(shouldThrow)
            , loc(l) { }

        /**
         * @brief Append text to the eventual exception/log message.
         *
         * @tparam U Streamable value type.
         * @param x Value to append.
         * @return Reference to this proxy.
         */
        template <typename U>
        ErrorIfProxy&
        operator<<(const U& x) {
            user_ss << x;
            return *this;
        }

        /**
         * @brief Finalize the error condition and optionally throw.
         *
         * @throws ExceptionT When @ref should_throw is `true`.
         *
         * @details
         * If throwing is enabled:
         * - a default message `"ATLAS error"` is used when the user did not stream text,
         * - logging is emitted at error severity when logging is enabled and active,
         * - `ExceptionT` is thrown with the final message.
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

    /**
     * @brief No-op proxy used when debug assertion macros are compiled out.
     *
     * @details
     * This type accepts stream input and discards it, preserving the syntax of:
     * @code
     * ATLAS_CHECK(cond) << "message";
     * @endcode
     *
     * in release builds without performing any work.
     */
    struct NoopErrorProxy final {

        /**
         * @brief Ignore streamed input and return the same proxy.
         *
         * @tparam U Arbitrary stream-like input type.
         * @param Unused value.
         * @return Reference to this proxy.
         */
        template <typename U>
        constexpr const NoopErrorProxy&
        operator<<(const U&) const noexcept {
            return *this;
        }
    };

    /**
     * @brief Singleton no-op proxy instance.
     */
    inline constexpr NoopErrorProxy noop_error_proxy {};

} // namespace detail

/**
 * @brief Create an error proxy that throws when @p cond is true.
 *
 * @tparam ExceptionT Exception type to throw. Defaults to `std::runtime_error`.
 * @param cond Trigger condition.
 * @param loc Source location of the call site.
 * @return Configured error proxy.
 *
 * @details
 * Intended for stream-style guarded error creation:
 * @code
 * atlas::error_if(x < 0) << "x must be non-negative";
 * @endcode
 */
template <typename ExceptionT = std::runtime_error>
inline detail::ErrorIfProxy<ExceptionT>
error_if(bool cond, std::source_location loc = std::source_location::current()) noexcept {
    return detail::ErrorIfProxy<ExceptionT>(cond, loc);
}

/**
 * @brief Create an error proxy that throws when @p cond is false.
 *
 * @tparam ExceptionT Exception type to throw. Defaults to `std::runtime_error`.
 * @param cond Condition expected to hold.
 * @param loc Source location of the call site.
 * @return Configured error proxy.
 *
 * @details
 * Intended for stream-style precondition/invariant checks:
 * @code
 * atlas::check(ptr != nullptr) << "pointer must not be null";
 * @endcode
 */
template <typename ExceptionT = std::runtime_error>
inline detail::ErrorIfProxy<ExceptionT>
check(bool cond, std::source_location loc = std::source_location::current()) noexcept {
    return detail::ErrorIfProxy<ExceptionT>(!cond, loc);
}

#if !defined(NDEBUG)
/**
 * @brief Debug-build macro wrapper for @ref atlas::error_if.
 *
 * @param cond Condition that triggers an exception when true.
 *
 * @details
 * In non-release builds, this expands to a throwing stream-capable proxy.
 */
#define ATLAS_ERROR_IF(cond) \
    ::atlas::error_if((cond), std::source_location::current())
#else
  /**
   * @brief Release-build no-op version of @ref ATLAS_ERROR_IF.
   *
   * @param cond Condition evaluated only for type correctness / unused suppression.
   *
   * @details
   * In release builds, this expands to a no-op proxy that discards streamed input.
   */
#define ATLAS_ERROR_IF(cond) \
    ((void)sizeof(cond), ::atlas::detail::noop_error_proxy)
#endif

#if !defined(NDEBUG)
/**
 * @brief Debug-build macro wrapper for @ref atlas::check.
 *
 * @param cond Condition expected to be true.
 *
 * @details
 * In non-release builds, this expands to a throwing stream-capable proxy
 * that triggers when the condition is false.
 */
#define ATLAS_CHECK(cond) \
    ::atlas::check((cond), std::source_location::current())
#else
  /**
   * @brief Release-build no-op version of @ref ATLAS_CHECK.
   *
   * @param cond Condition evaluated only for type correctness / unused suppression.
   *
   * @details
   * In release builds, this expands to a no-op proxy that discards streamed input.
   */
#define ATLAS_CHECK(cond) \
    ((void)sizeof(cond), ::atlas::detail::noop_error_proxy)
#endif

} // namespace atlas