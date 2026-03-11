#include <atlas/logging/logging.h>

#ifdef ATLAS_ENABLE_LOGGING

#include <chrono>
#include <ctime>
#include <iomanip>
#include <iostream>
#include <mutex>

namespace atlas::detail {

// ------------------------------------------------------------
// Global logging state (internal linkage)
//
// Purpose:
//   - Avoid static initialization order issues with iostreams
//   - Allow runtime configuration of streams and log level
//
// Policy:
//   - nullptr means fallback to default streams
//   - All accesses are protected by a mutex
// ------------------------------------------------------------
static std::ostream* s_info  = nullptr;
static std::ostream* s_warn  = nullptr;
static std::ostream* s_error = nullptr;
static std::ostream* s_debug = nullptr;

static LoggingLevel s_level = LoggingLevel::All;
static std::mutex s_mtx;

// ------------------------------------------------------------
// Compare logging levels using their underlying integer values
//
// Semantics:
//   a <= b means messages at level b are allowed when
//   the global level is set to a
// ------------------------------------------------------------
static inline bool is_leq(const LoggingLevel a, const LoggingLevel b) noexcept {
    return static_cast<uint8_t>(a) <= static_cast<uint8_t>(b);
}

// ------------------------------------------------------------
// Select output stream based on logging level
//
// Rules:
//   - If a custom stream is set, use it
//   - Otherwise fall back to default streams:
//       * Error  → std::cerr
//       * Others → std::cout
// ------------------------------------------------------------
static inline std::ostream* stream_for(const LoggingLevel level) {
    switch (level) {
    case LoggingLevel::Info:  return s_info  ? s_info  : &std::cout;
    case LoggingLevel::Warn:  return s_warn  ? s_warn  : &std::cout;
    case LoggingLevel::Error: return s_error ? s_error : &std::cerr;
    case LoggingLevel::Debug: return s_debug ? s_debug : &std::cout;
    default:                 return &std::cout;
    }
}

// ------------------------------------------------------------
// Convert logging level to a human-readable string
//
// Used in log line prefixes
// ------------------------------------------------------------
static inline const char* level_to_string(const LoggingLevel level) {
    switch (level) {
    case LoggingLevel::Info:  return "INFO";
    case LoggingLevel::Warn:  return "WARN";
    case LoggingLevel::Error: return "ERROR";
    case LoggingLevel::Debug: return "DEBUG";
    default:                 return "";
    }
}

// ------------------------------------------------------------
// Generate a formatted timestamp string
//
// Format:
//   YYYY-MM-DD HH:MM:SS
//
// Implementation notes:
//   - Uses thread-safe localtime variants per platform
// ------------------------------------------------------------
static inline std::string now_string() {
    using clock = std::chrono::system_clock;
    const auto t = clock::to_time_t(clock::now());
    std::tm tm{};
#if defined(_WIN32)
    localtime_s(&tm, &t);
#else
    localtime_r(&t, &tm);
#endif
    char buf[20];
    std::strftime(buf, sizeof(buf), "%F %T", &tm);
    return { buf };
}

// ------------------------------------------------------------
// Decide whether a message should be logged
//
// Rules:
//   - LoggingLevel::All allows all messages
//   - LoggingLevel::Off suppresses all messages
// ------------------------------------------------------------
bool should_log(const LoggingLevel msg_level) noexcept {
    return is_leq(s_level, msg_level);
}

// ------------------------------------------------------------
// Emit a single formatted log line
//
// Output format (IDE-clickable):
//   [LEVEL] timestamp file:line: function: message
//
// Guarantees:
//   - Thread-safe output
//   - Stream is flushed after each line
// ------------------------------------------------------------
void emit_log_line(const LoggingLevel msg_level,
                   const std::source_location& loc,
                   const std::string& message) {
    std::lock_guard<std::mutex> lock(s_mtx);
    if (!should_log(msg_level)) return;

    std::ostream* os = stream_for(msg_level);

    (*os) << "[" << level_to_string(msg_level) << "] "
          << now_string() << " "
          << loc.file_name() << ":" << loc.line() << ": "
          << loc.function_name() << ": "
          << message
          << std::endl;

    os->flush();
}

} // namespace atlas::detail


namespace atlas {

// ------------------------------------------------------------
// Logger destructor
//
// Design:
//   - RAII-based logging
//   - Message is accumulated via operator<<
//   - Log line is emitted upon destruction
// ------------------------------------------------------------
Logger::~Logger() {
    const std::string msg = _buffer.str();
    if (msg.empty()) return;

    if (::atlas::detail::should_log(_level)) {
        ::atlas::detail::emit_log_line(_level, _loc, msg);
    }
}

// ------------------------------------------------------------
// Stream configuration API
//
// Characteristics:
//   - Thread-safe
//   - Passing nullptr restores default stream behavior
// ------------------------------------------------------------
void Logging::set_info_stream(std::ostream* strm) {
    std::lock_guard<std::mutex> lock(detail::s_mtx);
    detail::s_info = strm;
}

void Logging::set_warn_stream(std::ostream* strm) {
    std::lock_guard<std::mutex> lock(detail::s_mtx);
    detail::s_warn = strm;
}

void Logging::set_error_stream(std::ostream* strm) {
    std::lock_guard<std::mutex> lock(detail::s_mtx);
    detail::s_error = strm;
}

void Logging::set_debug_stream(std::ostream* strm) {
    std::lock_guard<std::mutex> lock(detail::s_mtx);
    detail::s_debug = strm;
}

// ------------------------------------------------------------
// Assign the same stream to all log levels
// ------------------------------------------------------------
void Logging::set_all_stream(std::ostream* strm) {
    set_info_stream(strm);
    set_warn_stream(strm);
    set_error_stream(strm);
    set_debug_stream(strm);
}

// ------------------------------------------------------------
// Logging level control
// ------------------------------------------------------------
void Logging::set_level(LoggingLevel level) {
    std::lock_guard<std::mutex> lock(detail::s_mtx);
    detail::s_level = level;
}

void Logging::mute() {
    set_level(LoggingLevel::Off);
}

void Logging::unmute() {
    set_level(LoggingLevel::All);
}

} // namespace atlas

#endif // ATLAS_ENABLE_LOGGING
