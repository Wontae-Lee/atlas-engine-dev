#pragma once

#include <cstdint>
#include <source_location>
#include <sstream>
#include <stdexcept>
#include <string>

namespace atlas {

/**
 * @brief Severity threshold and per-message severity used by the logging system.
 *
 * The enumerators are ordered by increasing severity so that a single numeric
 * comparison decides whether a message passes the active threshold: a message
 * of level @p m is emitted when the configured threshold @p t satisfies
 * `t <= m` (see @ref should_log). Smaller therefore means "more verbose".
 *
 * The type serves two distinct roles:
 *  - as a *threshold* stored globally via @ref Logging::set_level, where
 *    @c all lets everything through and @c off suppresses everything;
 *  - as a *message level* attached to an individual log line (only
 *    @c debug, @c info, @c warn and @c error are ever used in this role).
 *
 * @note The fixed @c uint8_t underlying type is relied upon by the source file,
 *       which compares levels through `static_cast<uint8_t>`.
 */
enum class LoggingLevel : uint8_t {
    all   = 0, ///< Threshold that admits every message (most verbose).
    debug = 1, ///< Diagnostic detail; as a threshold admits debug and above.
    info  = 2, ///< Ordinary progress information.
    warn  = 3, ///< Recoverable anomaly worth surfacing.
    error = 4, ///< Failure condition; the highest ordinary message level.
    off   = 5  ///< Threshold that suppresses every message (least verbose).
};

// The whole active logging machinery below is compiled only when
// ATLAS_ENABLE_LOGGING is defined. The #else branch supplies zero-overhead
// stand-ins with identical call syntax so that call sites need no #ifdef guards.
#ifdef ATLAS_ENABLE_LOGGING

/**
 * @brief RAII accumulator that formats one log line and emits it on destruction.
 *
 * A @c Logger is meant to live only for the duration of a single streaming
 * expression such as `atlas::info() << "value = " << v;`. Every streamed
 * fragment is appended to an internal buffer; the completed message is written
 * out exactly once, when the temporary is destroyed at the end of the full
 * expression. Emission is skipped when nothing was streamed (empty buffer) or
 * when the message level does not pass the active threshold.
 *
 * The captured @ref std::source_location records the call site (file, line,
 * function) of the factory call, not of this destructor, so log lines point at
 * the user's code.
 *
 * @note Not intended to be stored or reused; copying is disabled and only the
 *       move operations exist so the factory helpers can return by value.
 * @warning Not thread-safe as an object: a single @c Logger must not be shared
 *          across threads. The final emission it performs, however, is
 *          serialized globally (see @ref emit_log_line).
 */
class Logger final {
public:
    /**
     * @brief Construct a logger bound to a message level and a call site.
     *
     * @param level Severity attached to the eventual log line.
     * @param loc   Source location of the originating call; defaulted so the
     *              factory helpers capture their own call site automatically.
     */
    explicit Logger(LoggingLevel level,
                    std::source_location loc = std::source_location::current()) noexcept
        : _level(level)
        , _loc(loc) { }

    /**
     * @brief Emit the accumulated message if it is non-empty and level-enabled.
     */
    ~Logger();

    /** @brief Copying is disabled; a logger owns a single pending log line. */
    Logger(const Logger&) = delete;
    /** @brief Copy assignment is disabled; see the deleted copy constructor. */
    Logger&
    operator=(const Logger&)
        = delete;

    /** @brief Move construction transfers the pending buffer and metadata. */
    Logger(Logger&&) noexcept = default;

    /** @brief Move assignment transfers the pending buffer and metadata. */
    Logger&
    operator=(Logger&&) noexcept = default;

    /**
     * @brief Append one value to the pending message.
     *
     * Forwards @p x to the internal @c std::stringstream, so any type with an
     * @c operator<< overload for @c std::ostream is accepted.
     *
     * @tparam U Type of the streamed value.
     * @param x Value to format and append.
     * @return Reference to this logger so streaming can be chained.
     * @note Declared @c const (and the buffer is @c mutable) so a temporary
     *       returned by value from @ref info et al. can still be streamed into.
     */
    template <typename U>
    const Logger&
    operator<<(const U& x) const {
        _buffer << x;
        return *this;
    }

private:
    LoggingLevel _level { LoggingLevel::info }; ///< Severity of the pending line.

    std::source_location _loc { std::source_location::current() }; ///< Originating call site.

    mutable std::stringstream _buffer {}; ///< Accumulated message text; mutable for const streaming.
};

/**
 * @brief Process-wide configuration facade for the logging system.
 *
 * All members are static and operate on file-scope global state defined in the
 * source: the per-level destination streams and the active severity threshold.
 * Every mutator takes the internal mutex, so configuration changes are safe to
 * call from any thread and stay synchronized with concurrent log emission.
 *
 * The class is never instantiated; it exists purely to namespace the setters.
 */
class Logging {
public:
    /**
     * @brief Override the destination stream for @c info messages.
     *
     * @param strm Stream to write @c info lines to; pass @c nullptr to restore
     *             the default (@c std::cout). The stream is not owned and must
     *             outlive any logging that targets it.
     */
    static void
    set_info_stream(std::ostream* strm);

    /**
     * @brief Override the destination stream for @c warn messages.
     *
     * @param strm Stream to write @c warn lines to; @c nullptr restores the
     *             default (@c std::cout). Not owned; must outlive its use.
     */
    static void
    set_warn_stream(std::ostream* strm);

    /**
     * @brief Override the destination stream for @c error messages.
     *
     * @param strm Stream to write @c error lines to; @c nullptr restores the
     *             default (@c std::cerr). Not owned; must outlive its use.
     */
    static void
    set_error_stream(std::ostream* strm);

    /**
     * @brief Override the destination stream for @c debug messages.
     *
     * @param strm Stream to write @c debug lines to; @c nullptr restores the
     *             default (@c std::cout). Not owned; must outlive its use.
     */
    static void
    set_debug_stream(std::ostream* strm);

    /**
     * @brief Point every ordinary message level at a single stream.
     *
     * Convenience wrapper that installs @p strm as the @c info, @c warn,
     * @c error and @c debug destination in one call.
     *
     * @param strm Shared destination stream; @c nullptr restores each level's
     *             own default. Not owned; must outlive its use.
     */
    static void
    set_all_stream(std::ostream* strm);

    /**
     * @brief Set the global severity threshold.
     *
     * A message of level @p m is emitted only while the threshold @p level
     * satisfies `level <= m`. @c LoggingLevel::all admits everything and
     * @c LoggingLevel::off suppresses everything.
     *
     * @param level New threshold applied to all subsequent logging.
     */
    static void
    set_level(LoggingLevel level);

    /** @brief Suppress all output by setting the threshold to @c off. */
    static void
    mute();

    /** @brief Restore full verbosity by setting the threshold to @c all. */
    static void
    unmute();
};

/**
 * @brief Begin an @c info-level log line.
 *
 * @param loc Call site, captured automatically by the default argument; do not
 *            pass it explicitly except to forward another location.
 * @return A temporary @ref Logger to stream the message into; it emits when the
 *         enclosing full expression ends.
 */
inline Logger
info(std::source_location loc = std::source_location::current()) {
    return Logger(LoggingLevel::info, loc);
}

/**
 * @brief Begin a @c warn-level log line.
 *
 * @param loc Call site, captured automatically by the default argument.
 * @return A temporary @ref Logger to stream the message into.
 */
inline Logger
warn(std::source_location loc = std::source_location::current()) {
    return Logger(LoggingLevel::warn, loc);
}

/**
 * @brief Begin an @c error-level log line.
 *
 * @param loc Call site, captured automatically by the default argument.
 * @return A temporary @ref Logger to stream the message into.
 */
inline Logger
error(std::source_location loc = std::source_location::current()) {
    return Logger(LoggingLevel::error, loc);
}

/**
 * @brief Begin a @c debug-level log line.
 *
 * @param loc Call site, captured automatically by the default argument.
 * @return A temporary @ref Logger to stream the message into.
 */
inline Logger
debug(std::source_location loc = std::source_location::current()) {
    return Logger(LoggingLevel::debug, loc);
}

/**
 * @brief Test whether a message level passes the current global threshold.
 *
 * @param msg_level Severity of the prospective message.
 * @return @c true if a message of @p msg_level would be emitted under the
 *         active threshold, @c false if it would be filtered out.
 * @note Thread-safe: reads the threshold atomically enough for a fast pre-check;
 *       @ref emit_log_line re-checks under the mutex to stay consistent.
 */
bool
should_log(LoggingLevel msg_level) noexcept;

/**
 * @brief Format and write one complete log line to its destination stream.
 *
 * Produces a line of the form
 * `[LEVEL] YYYY-MM-DD HH:MM:SS file:line: function: message`, chooses the
 * stream for @p msg_level, and flushes. The whole operation is serialized by a
 * global mutex so lines from different threads never interleave, and the level
 * is re-checked under that lock before anything is written.
 *
 * @param msg_level Severity that selects the tag and destination stream.
 * @param loc       Source location whose file, line and function are printed.
 * @param message   Fully assembled message body.
 */
void
emit_log_line(LoggingLevel msg_level,
              const std::source_location& loc,
              const std::string& message);

#else
// ATLAS_ENABLE_LOGGING is not defined: provide no-op replacements with the same
// spelling so that logging call sites compile away to nothing.

/**
 * @brief Stand-in for @ref Logger used when logging is compiled out.
 *
 * Every streamed value is discarded. The template @c operator<< accepts any
 * argument and returns @c *this so that a chained expression such as
 * `atlas::info() << a << b;` still compiles and, after inlining, generates no
 * code. Streamed arguments are still evaluated by the language (their side
 * effects run); only the formatting and output are elided.
 */
struct NullLogger {

    /**
     * @brief Discard one streamed value and continue the chain.
     * @tparam U Type of the ignored value.
     * @return Reference to this sink so streaming can be chained.
     */
    template <typename U>
    const NullLogger&
    operator<<(const U&) const {
        return *this;
    }
};

/// Shared immutable sink instance handed out by @ref make_null_logger.
inline constexpr NullLogger atlas_null_logger {};

/**
 * @brief Return the shared @ref NullLogger sink.
 *
 * @return Reference to the process-wide @ref atlas_null_logger. Constexpr and
 *         allocation-free so the disabled logging factories cost nothing.
 */
constexpr const NullLogger&
make_null_logger() noexcept {
    return atlas_null_logger;
}

/**
 * @brief No-op counterpart of the active @ref Logging facade.
 *
 * Mirrors the enabled interface exactly, but every member is an empty inline
 * function, so configuration calls disappear when logging is compiled out.
 */
class Logging {
public:
    /** @brief No-op; accepts and ignores an @c info stream override. */
    static void
    set_info_stream(std::ostream*) { }

    /** @brief No-op; accepts and ignores a @c warn stream override. */
    static void
    set_warn_stream(std::ostream*) { }

    /** @brief No-op; accepts and ignores an @c error stream override. */
    static void
    set_error_stream(std::ostream*) { }

    /** @brief No-op; accepts and ignores a @c debug stream override. */
    static void
    set_debug_stream(std::ostream*) { }

    /** @brief No-op; accepts and ignores an all-levels stream override. */
    static void
    set_all_stream(std::ostream*) { }

    /** @brief No-op; accepts and ignores a threshold change. */
    static void
    set_level(LoggingLevel) { }

    /** @brief No-op; there is nothing to mute. */
    static void
    mute() { }

    /** @brief No-op; there is nothing to unmute. */
    static void
    unmute() { }
};

/**
 * @brief Disabled-build counterpart of @ref info; returns the shared sink.
 * @return Reference to the shared @ref NullLogger.
 */
inline const NullLogger&
info(std::source_location = std::source_location::current()) {
    return make_null_logger();
}

/**
 * @brief Disabled-build counterpart of @ref warn; returns the shared sink.
 * @return Reference to the shared @ref NullLogger.
 */
inline const NullLogger&
warn(std::source_location = std::source_location::current()) {
    return make_null_logger();
}

/**
 * @brief Disabled-build counterpart of @ref error; returns the shared sink.
 * @return Reference to the shared @ref NullLogger.
 */
inline const NullLogger&
error(std::source_location = std::source_location::current()) {
    return make_null_logger();
}

/**
 * @brief Disabled-build counterpart of @ref debug; returns the shared sink.
 * @return Reference to the shared @ref NullLogger.
 */
inline const NullLogger&
debug(std::source_location = std::source_location::current()) {
    return make_null_logger();
}

#endif

/**
 * @brief RAII proxy that optionally logs and throws when a condition fails.
 *
 * Returned by @ref error_if / @ref check, this proxy collects an optional
 * explanatory message via @c operator<< and, if it was armed to throw, performs
 * the throw from its destructor at the end of the full expression. This lets a
 * check read as a single streaming statement:
 * `atlas::check(ptr != nullptr) << "null handle for " << name;`.
 *
 * When it throws it first emits an @c error log line (only in builds where
 * @c ATLAS_ENABLE_LOGGING is defined); the throw itself happens regardless of
 * whether logging is compiled in.
 *
 * @tparam ExceptionT Exception type constructed from the message and thrown.
 *
 * @warning The destructor is @c noexcept(false) and throws by design. Do not
 *          create one of these while an exception is already propagating, as
 *          throwing during unwinding calls @c std::terminate.
 */
template <typename ExceptionT>
struct ErrorIfProxy final {

    bool should_throw { false }; ///< Whether the destructor will throw.

    std::source_location loc { std::source_location::current() }; ///< Site to report.

    std::stringstream user_ss {}; ///< Accumulated user message.

    /**
     * @brief Arm the proxy and record the failing call site.
     *
     * @param shouldThrow @c true to throw on destruction, @c false to do nothing.
     * @param l           Source location blamed in the log line and available to
     *                    the caller; defaulted to the construction site.
     */
    explicit ErrorIfProxy(bool shouldThrow,
                          std::source_location l = std::source_location::current()) noexcept
        : should_throw(shouldThrow)
        , loc(l) { }

    /**
     * @brief Append one value to the pending failure message.
     * @tparam U Type of the streamed value.
     * @param x Value to format and append.
     * @return Reference to this proxy so streaming can be chained.
     */
    template <typename U>
    ErrorIfProxy&
    operator<<(const U& x) {
        user_ss << x;
        return *this;
    }

    /**
     * @brief Log and throw @c ExceptionT if the proxy was armed.
     *
     * Does nothing when @ref should_throw is @c false. Otherwise it builds the
     * message (substituting @c "ATLAS error" when none was streamed), emits an
     * @c error log line where logging is enabled, and throws.
     *
     * @throws ExceptionT constructed from the assembled message when armed.
     */
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

/**
 * @brief Zero-cost stand-in returned by the check macros in release builds.
 *
 * Discards any streamed message and never throws, so that `ATLAS_CHECK(x) << m`
 * compiles to nothing when checks are disabled by @c NDEBUG.
 */
struct NoopErrorProxy final {

    /**
     * @brief Discard one streamed value and continue the chain.
     * @tparam U Type of the ignored value.
     * @return Reference to this no-op proxy.
     */
    template <typename U>
    constexpr const NoopErrorProxy&
    operator<<(const U&) const noexcept {
        return *this;
    }
};

/// Shared immutable no-op proxy used by the release-build check macros.
inline constexpr NoopErrorProxy noop_error_proxy {};

/**
 * @brief Throw @c ExceptionT when @p cond is @c true.
 *
 * @tparam ExceptionT Exception to throw; defaults to @c std::runtime_error.
 * @param cond Condition that, when @c true, arms the returned proxy to throw.
 * @param loc  Call site, captured automatically by the default argument.
 * @return An @ref ErrorIfProxy; stream an explanatory message into it, and the
 *         throw (if armed) fires when the full expression ends.
 * @note This factory is itself @c noexcept; the throw happens later, in the
 *       proxy destructor.
 */
template <typename ExceptionT = std::runtime_error>
inline ErrorIfProxy<ExceptionT>
error_if(bool cond, std::source_location loc = std::source_location::current()) noexcept {
    return ErrorIfProxy<ExceptionT>(cond, loc);
}

/**
 * @brief Throw @c ExceptionT when @p cond is @c false (assertion form).
 *
 * The logical inverse of @ref error_if: it arms the proxy when the asserted
 * condition does not hold.
 *
 * @tparam ExceptionT Exception to throw; defaults to @c std::runtime_error.
 * @param cond Condition expected to hold; a @c false value arms the throw.
 * @param loc  Call site, captured automatically by the default argument.
 * @return An @ref ErrorIfProxy to stream an explanatory message into.
 */
template <typename ExceptionT = std::runtime_error>
inline ErrorIfProxy<ExceptionT>
check(bool cond, std::source_location loc = std::source_location::current()) noexcept {
    return ErrorIfProxy<ExceptionT>(!cond, loc);
}

// ATLAS_ERROR_IF / ATLAS_CHECK compile to real checks only in debug builds
// (NDEBUG undefined). In release builds they expand to the no-op proxy, and the
// `(void)sizeof(cond)` keeps `cond` type-checked while leaving it unevaluated.

#if !defined(NDEBUG)

/**
 * @brief Debug-only guard that throws (and logs) when @p cond is @c true.
 * @param cond Failure condition; evaluated only in debug builds.
 * @warning In release builds @p cond is not evaluated, so it must be free of
 *          side effects the program relies on.
 */
#define ATLAS_ERROR_IF(cond) \
    ::atlas::error_if((cond), std::source_location::current())
#else

/**
 * @brief Release-build expansion of @ref ATLAS_ERROR_IF: a no-op proxy.
 * @param cond Condition, type-checked via @c sizeof but not evaluated.
 */
#define ATLAS_ERROR_IF(cond) \
    ((void)sizeof(cond), ::atlas::noop_error_proxy)
#endif

#if !defined(NDEBUG)

/**
 * @brief Debug-only assertion that throws (and logs) when @p cond is @c false.
 * @param cond Condition expected to hold; evaluated only in debug builds.
 * @warning In release builds @p cond is not evaluated, so it must be free of
 *          side effects the program relies on.
 */
#define ATLAS_CHECK(cond) \
    ::atlas::check((cond), std::source_location::current())
#else

/**
 * @brief Release-build expansion of @ref ATLAS_CHECK: a no-op proxy.
 * @param cond Condition, type-checked via @c sizeof but not evaluated.
 */
#define ATLAS_CHECK(cond) \
    ((void)sizeof(cond), ::atlas::noop_error_proxy)
#endif

}