#include <atlas/logging/logging.h>

#ifdef ATLAS_ENABLE_LOGGING

#include <chrono>
#include <ctime>
#include <iomanip>
#include <iostream>
#include <mutex>

namespace atlas::detail {

static std::ostream* s_info  = nullptr;
static std::ostream* s_warn  = nullptr;
static std::ostream* s_error = nullptr;
static std::ostream* s_debug = nullptr;

static LoggingLevel s_level = LoggingLevel::All;
static std::mutex s_mtx;

static inline bool
is_leq(const LoggingLevel a, const LoggingLevel b) noexcept {
    return static_cast<uint8_t>(a) <= static_cast<uint8_t>(b);
}

static inline std::ostream*
stream_for(const LoggingLevel level) {
    switch (level) {
    case LoggingLevel::Info:
        return s_info ? s_info : &std::cout;
    case LoggingLevel::Warn:
        return s_warn ? s_warn : &std::cout;
    case LoggingLevel::Error:
        return s_error ? s_error : &std::cerr;
    case LoggingLevel::Debug:
        return s_debug ? s_debug : &std::cout;
    default:
        return &std::cout;
    }
}

static inline const char*
level_to_string(const LoggingLevel level) {
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
        return "";
    }
}

static inline std::string
now_string() {
    using clock  = std::chrono::system_clock;
    const auto t = clock::to_time_t(clock::now());
    std::tm tm {};
#if defined(_WIN32)
    localtime_s(&tm, &t);
#else
    localtime_r(&t, &tm);
#endif
    char buf[20];
    std::strftime(buf, sizeof(buf), "%F %T", &tm);
    return { buf };
}

bool
should_log(const LoggingLevel msg_level) noexcept {
    return is_leq(s_level, msg_level);
}

void
emit_log_line(const LoggingLevel msg_level,
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

}

namespace atlas {

Logger::~Logger() {
    const std::string msg = _buffer.str();
    if (msg.empty()) return;

    if (::atlas::detail::should_log(_level)) {
        ::atlas::detail::emit_log_line(_level, _loc, msg);
    }
}

void
Logging::set_info_stream(std::ostream* strm) {
    std::lock_guard<std::mutex> lock(detail::s_mtx);
    detail::s_info = strm;
}

void
Logging::set_warn_stream(std::ostream* strm) {
    std::lock_guard<std::mutex> lock(detail::s_mtx);
    detail::s_warn = strm;
}

void
Logging::set_error_stream(std::ostream* strm) {
    std::lock_guard<std::mutex> lock(detail::s_mtx);
    detail::s_error = strm;
}

void
Logging::set_debug_stream(std::ostream* strm) {
    std::lock_guard<std::mutex> lock(detail::s_mtx);
    detail::s_debug = strm;
}

void
Logging::set_all_stream(std::ostream* strm) {
    set_info_stream(strm);
    set_warn_stream(strm);
    set_error_stream(strm);
    set_debug_stream(strm);
}

void
Logging::set_level(LoggingLevel level) {
    std::lock_guard<std::mutex> lock(detail::s_mtx);
    detail::s_level = level;
}

void
Logging::mute() {
    set_level(LoggingLevel::Off);
}

void
Logging::unmute() {
    set_level(LoggingLevel::All);
}

}

#endif