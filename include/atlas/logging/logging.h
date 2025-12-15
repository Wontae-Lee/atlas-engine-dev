#pragma once

#include <cstdint>
#include <ostream>
#include <sstream>
#include <string>

namespace atlas {
enum class LoggingLevel : uint8_t {
    All = 0,
    Debug = 1,
    Info = 2,
    Warn = 3,
    Error = 4,
    Off = 5
};

#ifdef ATLAS_ENABLE_LOGGING

class Logger final {
public:
    explicit
    Logger(LoggingLevel level)
        : _level(level) {}

    ~Logger();

    template <typename T>
    const Logger&
    operator<<(const T& x) const {
        _buffer << x;
        return *this;
    }

private:
    LoggingLevel _level;
    mutable std::stringstream _buffer;
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

    static std::string
    get_header(LoggingLevel level);

    static void
    set_level(LoggingLevel level);
    static void
    mute();
    static void
    unmute();
};

#else

struct NullLogger {
    template <typename T>
    const NullLogger&
    operator<<(const T&) const { return *this; }
};

inline constexpr NullLogger atlas_null_logger{};

constexpr const NullLogger&
make_null_logger() {
    return static_cast<const NullLogger&>(atlas_null_logger);
}

class Logging {
public:
    static void
    set_info_stream(std::ostream*) {}

    static void
    set_warn_stream(std::ostream*) {}

    static void
    set_error_stream(std::ostream*) {}

    static void
    set_debug_stream(std::ostream*) {}

    static void
    set_all_stream(std::ostream*) {}

    static std::string
    get_header(LoggingLevel) { return {}; }

    static void
    set_level(LoggingLevel) {}

    static void
    mute() {}

    static void
    unmute() {}
};

#endif
}


#ifdef ATLAS_ENABLE_LOGGING
#define ATLAS_INFO  (::atlas::Logger(::atlas::LoggingLevel::Info)                  \
                     << ::atlas::Logging::get_header(::atlas::LoggingLevel::Info)  \
                     << "[" << __FILE__ << ":" << __LINE__ << " (" << __func__ << ")] ")
#define ATLAS_WARN  (::atlas::Logger(::atlas::LoggingLevel::Warn)                  \
                     << ::atlas::Logging::get_header(::atlas::LoggingLevel::Warn)  \
                     << "[" << __FILE__ << ":" << __LINE__ << " (" << __func__ << ")] ")
#define ATLAS_ERROR (::atlas::Logger(::atlas::LoggingLevel::Error)                 \
                     << ::atlas::Logging::get_header(::atlas::LoggingLevel::Error) \
                     << "[" << __FILE__ << ":" << __LINE__ << " (" << __func__ << ")] ")
#define ATLAS_DEBUG (::atlas::Logger(::atlas::LoggingLevel::Debug)                 \
                     << ::atlas::Logging::get_header(::atlas::LoggingLevel::Debug) \
                     << "[" << __FILE__ << ":" << __LINE__ << " (" << __func__ << ")] ")
#else
#define ATLAS_INFO  (::atlas::make_null_logger())
#define ATLAS_WARN  (::atlas::make_null_logger())
#define ATLAS_ERROR (::atlas::make_null_logger())
#define ATLAS_DEBUG (::atlas::make_null_logger())
#endif


#ifdef ATLAS_ENABLE_LOGGING

#include <chrono>
#include <ctime>
#include <iostream>
#include <mutex>
#include <stdexcept>
#include <utility>

namespace atlas::detail {
inline std::mutex critical;

inline std::ostream* info_out_stream  = &std::cout;
inline std::ostream* warn_out_stream  = &std::cout;
inline std::ostream* error_out_stream = &std::cerr;
inline std::ostream* debug_out_stream = &std::cout;

inline LoggingLevel s_logging_level = LoggingLevel::All;

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

inline const char*
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


template <typename ExceptionT>
[[noreturn]] inline void
log_and_throw_impl_typed(LoggingLevel level, std::string msg) {
    {
        std::lock_guard<std::mutex> lock(critical);
        if (is_leq(s_logging_level, level)) {
            auto* strm = level_to_stream(level);
            (*strm) << msg << std::endl;
            strm->flush();
        }
    }
    throw ExceptionT(std::move(msg));
}


template <typename ExceptionT>
struct ErrorIfProxyT {
    const char* file;
    int line;
    const char* func;

    std::stringstream user_ss;

    ErrorIfProxyT(const char* f, int l, const char* fn)
        : file(f), line(l), func(fn) {}

    template <typename T>
    ErrorIfProxyT&
    operator<<(const T& x) {
        user_ss << x;
        return *this;
    }

    ~ErrorIfProxyT() noexcept(false) {
        const std::string user_msg = user_ss.str();

        std::stringstream log_ss;
        log_ss << ::atlas::Logging::get_header(::atlas::LoggingLevel::Error)
            << "[" << file << ":" << line << " (" << func << ")] "
            << user_msg;

        {
            std::lock_guard<std::mutex> lock(critical);
            if (is_leq(s_logging_level, LoggingLevel::Error)) {
                auto* strm = level_to_stream(LoggingLevel::Error);
                (*strm) << log_ss.str() << std::endl;
                strm->flush();
            }
        }
        throw ExceptionT(user_msg);
    }
};

}

namespace atlas {
inline
Logger::~Logger() {
    std::lock_guard<std::mutex> lock(detail::critical);
    if (detail::is_leq(detail::s_logging_level, _level)) {
        auto* strm = detail::level_to_stream(_level);
        (*strm) << _buffer.str() << std::endl;
        strm->flush();
    }
}

inline void
Logging::set_info_stream(std::ostream* strm) {
    std::lock_guard<std::mutex> lock(detail::critical);
    detail::info_out_stream = strm;
}

inline void
Logging::set_warn_stream(std::ostream* strm) {
    std::lock_guard<std::mutex> lock(detail::critical);
    detail::warn_out_stream = strm;
}

inline void
Logging::set_error_stream(std::ostream* strm) {
    std::lock_guard<std::mutex> lock(detail::critical);
    detail::error_out_stream = strm;
}

inline void
Logging::set_debug_stream(std::ostream* strm) {
    std::lock_guard<std::mutex> lock(detail::critical);
    detail::debug_out_stream = strm;
}

inline void
Logging::set_all_stream(std::ostream* strm) {
    set_info_stream(strm);
    set_warn_stream(strm);
    set_error_stream(strm);
    set_debug_stream(strm);
}

inline std::string
Logging::get_header(LoggingLevel level) {
    using clock = std::chrono::system_clock;
    auto now    = clock::to_time_t(clock::now());

    char timeStr[20];
    std::strftime(timeStr, sizeof(timeStr), "%F %T", std::localtime(&now));

    char header[256];
    std::snprintf(header,
                  sizeof(header),
                  "[%s] %s ",
                  detail::level_to_string(level),
                  timeStr);
    return header;
}

inline void
Logging::set_level(LoggingLevel level) {
    std::lock_guard<std::mutex> lock(detail::critical);
    detail::s_logging_level = level;
}

inline void
Logging::mute() {
    set_level(LoggingLevel::Off);
}

inline void
Logging::unmute() {
    set_level(LoggingLevel::All);
}
}


#define ATLAS_ERROR_IF_EX(cond, ExceptionType) \
if (!(cond)) {} else ::atlas::detail::ErrorIfProxyT<ExceptionType>{__FILE__, __LINE__, __func__}

#define ATLAS_ERROR_IF(cond) ATLAS_ERROR_IF_EX((cond), std::runtime_error)
#define ATLAS_CHECK_EX(cond, ExceptionType) ATLAS_ERROR_IF_EX(!(cond), ExceptionType)
#define ATLAS_CHECK(cond) ATLAS_CHECK_EX((cond), std::runtime_error)


#define ATLAS_THROW_ERROR_EX(ExceptionType) ATLAS_ERROR_IF_EX(true, ExceptionType)
#define ATLAS_THROW_ERROR()                ATLAS_ERROR_IF(true)

#else

#include <stdexcept>
#include <string>


namespace atlas::detail {
template <typename ExceptionT>
struct ErrorIfProxyNoLogT {
    std::stringstream ss;

    template <typename T>
    ErrorIfProxyNoLogT&
    operator<<(const T& x) {
        ss << x;
        return *this;
    }

    ~ErrorIfProxyNoLogT() noexcept(false) {
        const std::string msg = ss.str().empty() ? "ATLAS error" : ss.str();
        throw ExceptionT(msg);
    }
};
}

#define ATLAS_ERROR_IF_EX(cond, ExceptionType) \
    if (!(cond)) {} else ::atlas::detail::ErrorIfProxyNoLogT<ExceptionType>{}

#define ATLAS_ERROR_IF(cond) ATLAS_ERROR_IF_EX((cond), std::runtime_error)

#define ATLAS_CHECK_EX(cond, ExceptionType) ATLAS_ERROR_IF_EX(!(cond), ExceptionType)
#define ATLAS_CHECK(cond)                  ATLAS_CHECK_EX((cond), std::runtime_error)

#define ATLAS_THROW_ERROR_EX(ExceptionType) ATLAS_ERROR_IF_EX(true, ExceptionType)
#define ATLAS_THROW_ERROR()                ATLAS_ERROR_IF(true)

#endif