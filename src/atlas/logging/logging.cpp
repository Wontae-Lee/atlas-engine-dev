#include <atlas/logging/logging.h>

#ifdef ATLAS_ENABLE_LOGGING

#include <chrono>
#include <cstdlib>
#include <exception>
#include <fstream>
#include <iostream>
#include <mutex>
#include <string>

namespace atlas {


static std::mutex   critical;
static std::ostream* info_out_stream  = &std::cout;
static std::ostream* warn_out_stream  = &std::cout;
static std::ostream* error_out_stream = &std::cerr;
static std::ostream* debug_out_stream = &std::cout;
static LoggingLevel  s_logging_level  = LoggingLevel::All;

inline std::ostream*
level_to_stream(LoggingLevel level) {
    switch (level) {
    case LoggingLevel::Info:
        return info_out_stream;
    case LoggingLevel::Warn:
        return warn_out_stream;
    case LoggingLevel::Error:
        return error_out_stream;
    case LoggingLevel::Debug:
        return debug_out_stream;
    default:
        return info_out_stream;
    }
}

inline std::string
level_to_string(LoggingLevel level) {
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

inline bool
is_leq(LoggingLevel a, LoggingLevel b) {
    return static_cast<uint8_t>(a) <= static_cast<uint8_t>(b);
}

Logger::Logger(LoggingLevel level)
    : _level(level) { }

Logger::~Logger() {
    std::lock_guard<std::mutex> lock(critical);
    if (is_leq(s_logging_level, _level)) {
        auto strm = level_to_stream(_level);
        (*strm) << _buffer.str() << std::endl;
        strm->flush();
    }
}

void
Logging::set_info_stream(std::ostream* strm) {
    std::lock_guard<std::mutex> lock(critical);
    info_out_stream = strm;
}

void
Logging::set_warn_stream(std::ostream* strm) {
    std::lock_guard<std::mutex> lock(critical);
    warn_out_stream = strm;
}

void
Logging::set_error_stream(std::ostream* strm) {
    std::lock_guard<std::mutex> lock(critical);
    error_out_stream = strm;
}

void
Logging::set_debug_stream(std::ostream* strm) {
    std::lock_guard<std::mutex> lock(critical);
    debug_out_stream = strm;
}

void
Logging::set_all_stream(std::ostream* strm) {
    set_info_stream(strm);
    set_warn_stream(strm);
    set_error_stream(strm);
    set_debug_stream(strm);
}

std::string
Logging::get_header(LoggingLevel level) {
    auto now = std::chrono::system_clock::to_time_t(std::chrono::system_clock::now());
    char timeStr[20];
    std::strftime(timeStr, sizeof(timeStr), "%F %T", std::localtime(&now));
    char header[256];
    std::snprintf(header, sizeof(header), "[%s] %s ",
                  level_to_string(level).c_str(), timeStr);
    return header;
}

void
Logging::set_level(LoggingLevel level) {
    std::lock_guard<std::mutex> lock(critical);
    s_logging_level = level;
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