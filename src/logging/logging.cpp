#include <atlas/logging/logging.h>

#ifdef ATLAS_ENABLE_LOGGING

#include <chrono>
#include <ctime>
#include <iomanip>
#include <iostream>
#include <mutex>

namespace atlas::detail {

// Optional output stream override for INFO messages.
// If null, INFO logs are written to std::cout.
static std::ostream* s_info = nullptr;

// Optional output stream override for WARN messages.
// If null, WARN logs are written to std::cout.
static std::ostream* s_warn = nullptr;

// Optional output stream override for ERROR messages.
// If null, ERROR logs are written to std::cerr.
static std::ostream* s_error = nullptr;

// Optional output stream override for DEBUG messages.
// If null, DEBUG logs are written to std::cout.
static std::ostream* s_debug = nullptr;

// Global logging threshold.
//
// Interpretation:
/// - LoggingLevel::All   : allow all log messages
/// - LoggingLevel::Debug : allow Debug, Info, Warn, Error
/// - LoggingLevel::Info  : allow Info, Warn, Error
/// - LoggingLevel::Warn  : allow Warn, Error
/// - LoggingLevel::Error : allow Error only
/// - LoggingLevel::Off   : allow nothing
static LoggingLevel s_level = LoggingLevel::All;

// Global mutex protecting:
// - stream pointer updates
// - level changes
// - final log line emission
//
// This ensures log lines are not interleaved across threads and configuration
// changes remain synchronized with logging activity.
static std::mutex s_mtx;

static inline bool
is_leq(const LoggingLevel a, const LoggingLevel b) noexcept {
    // Compare logging levels by their underlying numeric severity ordering.
    //
    // The enum is arranged so that "smaller" means "more verbose":
    //   All < Debug < Info < Warn < Error < Off
    //
    // This helper is used to answer:
    //   "Does current threshold a allow message level b?"
    return static_cast<uint8_t>(a) <= static_cast<uint8_t>(b);
}

static inline std::ostream*
stream_for(const LoggingLevel level) {
    // Select the output stream associated with a given message level.
    //
    // Behavior:
    // - use caller-configured stream if one was explicitly installed
    // - otherwise fall back to a sensible default stream
    switch (level) {
    case LoggingLevel::Info:
        // INFO defaults to stdout.
        return s_info ? s_info : &std::cout;

    case LoggingLevel::Warn:
        // WARN defaults to stdout.
        return s_warn ? s_warn : &std::cout;

    case LoggingLevel::Error:
        // ERROR defaults to stderr.
        return s_error ? s_error : &std::cerr;

    case LoggingLevel::Debug:
        // DEBUG defaults to stdout.
        return s_debug ? s_debug : &std::cout;

    default:
        // Any unexpected level falls back to stdout.
        return &std::cout;
    }
}

static inline const char*
level_to_string(const LoggingLevel level) {
    // Convert a logging level to the short string tag written into each log line.
    switch (level) {
    case LoggingLevel::Info:
        return "INFO";

    case LoggingLevel::Warn:
        return "WARN";

    case LoggingLevel::Error:
        return "ERROR";

    case LoggingLevel::Debug:
        return "DEBUG";

    default:
        // For levels that are not emitted as ordinary message tags
        // (such as All / Off), return an empty string.
        return "";
    }
}

static inline std::string
now_string() {
    // Build a local timestamp string in the format:
    //   YYYY-MM-DD HH:MM:SS
    //
    // Steps:
    // 1. read current system clock time
    // 2. convert to calendar time
    // 3. convert to thread-safe local time representation
    // 4. format into a fixed-size buffer
    using clock  = std::chrono::system_clock;
    const auto t = clock::to_time_t(clock::now());

    std::tm tm {};

#if defined(_WIN32)
    // Windows thread-safe local time conversion.
    localtime_s(&tm, &t);
#else
    // POSIX thread-safe local time conversion.
    localtime_r(&t, &tm);
#endif

    char buf[20];

    // "%F %T" expands to:
    // - %F : YYYY-MM-DD
    // - %T : HH:MM:SS
    std::strftime(buf, sizeof(buf), "%F %T", &tm);

    return { buf };
}

bool
should_log(const LoggingLevel msg_level) noexcept {
    // Return whether the current global logging threshold allows a message
    // of the given level to be emitted.
    //
    // Example:
    // - s_level == Info  -> Info/Warn/Error allowed, Debug filtered out
    // - s_level == Off   -> everything filtered out
    return is_leq(s_level, msg_level);
}

void
emit_log_line(const LoggingLevel msg_level,
              const std::source_location& loc,
              const std::string& message) {
    // Emit one fully formatted log line in a thread-safe way.
    //
    // Output format:
    //   [LEVEL] YYYY-MM-DD HH:MM:SS file:line: function: message
    //
    // Lock scope covers:
    // - threshold check
    // - stream lookup
    // - formatted write
    // - flush
    //
    // This prevents concurrent threads from interleaving partial log lines.
    std::lock_guard<std::mutex> lock(s_mtx);

    // Re-check the logging threshold under the mutex so the decision is
    // consistent with the current synchronized configuration state.
    if (!should_log(msg_level)) return;

    // Select the destination stream for this message level.
    std::ostream* os = stream_for(msg_level);

    // Write one complete, human-readable log line with contextual metadata.
    (*os) << "[" << level_to_string(msg_level) << "] "
          << now_string() << " "
          << loc.file_name() << ":" << loc.line() << ": "
          << loc.function_name() << ": "
          << message
          << std::endl;

    // Flush immediately so log output is visible promptly and less likely
    // to be lost in buffered streams on abnormal termination.
    os->flush();
}

} // namespace atlas::detail

namespace atlas {

Logger::~Logger() {
    // Finalize deferred log message emission.
    //
    // Logger accumulates streamed fragments into _buffer during its lifetime.
    // On destruction:
    // - if the message is empty, do nothing
    // - otherwise, check the current logging threshold
    // - emit the completed line with source-location metadata
    //
    // This RAII pattern enables call sites such as:
    //   atlas::info() << "hello " << value;
    const std::string msg = _buffer.str();

    // Skip emission when nothing was streamed into the logger.
    if (msg.empty()) return;

    // Only emit if the message level is currently enabled.
    if (::atlas::detail::should_log(_level)) {
        ::atlas::detail::emit_log_line(_level, _loc, msg);
    }
}

void
Logging::set_info_stream(std::ostream* strm) {
    // Install or replace the output stream used for INFO messages.
    //
    // Passing nullptr restores default fallback behavior (std::cout).
    std::lock_guard<std::mutex> lock(detail::s_mtx);
    detail::s_info = strm;
}

void
Logging::set_warn_stream(std::ostream* strm) {
    // Install or replace the output stream used for WARN messages.
    //
    // Passing nullptr restores default fallback behavior (std::cout).
    std::lock_guard<std::mutex> lock(detail::s_mtx);
    detail::s_warn = strm;
}

void
Logging::set_error_stream(std::ostream* strm) {
    // Install or replace the output stream used for ERROR messages.
    //
    // Passing nullptr restores default fallback behavior (std::cerr).
    std::lock_guard<std::mutex> lock(detail::s_mtx);
    detail::s_error = strm;
}

void
Logging::set_debug_stream(std::ostream* strm) {
    // Install or replace the output stream used for DEBUG messages.
    //
    // Passing nullptr restores default fallback behavior (std::cout).
    std::lock_guard<std::mutex> lock(detail::s_mtx);
    detail::s_debug = strm;
}

void
Logging::set_all_stream(std::ostream* strm) {
    // Install the same stream for all ordinary message levels:
    // - INFO
    // - WARN
    // - ERROR
    // - DEBUG
    //
    // This is implemented by delegating to the per-level setters.
    set_info_stream(strm);
    set_warn_stream(strm);
    set_error_stream(strm);
    set_debug_stream(strm);
}

void
Logging::set_level(LoggingLevel level) {
    // Replace the global logging threshold.
    //
    // This affects future calls to should_log() and therefore controls which
    // messages are emitted after this point.
    std::lock_guard<std::mutex> lock(detail::s_mtx);
    detail::s_level = level;
}

void
Logging::mute() {
    // Disable all log emission globally.
    set_level(LoggingLevel::Off);
}

void
Logging::unmute() {
    // Restore the most verbose global logging mode.
    set_level(LoggingLevel::All);
}

} // namespace atlas

#endif