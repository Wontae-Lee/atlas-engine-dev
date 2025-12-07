#ifndef INCLUDE_ATLAS_CORE_LOGGING_H
#define INCLUDE_ATLAS_CORE_LOGGING_H

#include <cstdint>
#include <sstream>
#include <string>
#include <ostream>

namespace atlas {

enum class LoggingLevel : uint8_t {
    All   = 0,
    Debug = 1,
    Info  = 2,
    Warn  = 3,
    Error = 4,
    Off   = 5
};

#ifdef ATLAS_ENABLE_LOGGING

class Logger final {
public:
    explicit Logger(LoggingLevel level);
    ~Logger();

    template <typename T>
    const Logger& operator<<(const T& x) const {
        _buffer << x;
        return *this;
    }

private:
    LoggingLevel _level;
    mutable std::stringstream _buffer;
};

class Logging {
public:
    static void set_info_stream(std::ostream* strm);
    static void set_warn_stream(std::ostream* strm);
    static void set_error_stream(std::ostream* strm);
    static void set_debug_stream(std::ostream* strm);
    static void set_all_stream(std::ostream* strm);
    static std::string get_header(LoggingLevel level);
    static void set_level(LoggingLevel level);
    static void mute();
    static void unmute();
};







#else



struct NullLogger {
    template <typename T>
    const NullLogger& operator<<(const T&) const { return *this; }
};
inline constexpr NullLogger atlas_null_logger{};
constexpr const NullLogger& make_null_logger() {
    return static_cast<const NullLogger&>(atlas_null_logger);
}


class Logging {
public:
    static void set_info_stream(std::ostream*) {}
    static void set_warn_stream(std::ostream*) {}
    static void set_error_stream(std::ostream*) {}
    static void set_debug_stream(std::ostream*) {}
    static void set_all_stream(std::ostream*) {}
    static std::string get_header(LoggingLevel) { return {}; }
    static void set_level(LoggingLevel) {}
    static void mute() {}
    static void unmute() {}
};

#endif

}



#ifdef ATLAS_ENABLE_LOGGING

#define ATLAS_INFO  ( ::atlas::Logger(::atlas::LoggingLevel::Info)  \
                     << ::atlas::Logging::get_header(::atlas::LoggingLevel::Info)  \
                     << "[" << __FILE__ << ":" << __LINE__ << " (" << __func__ << ")] " )

#define ATLAS_WARN  ( ::atlas::Logger(::atlas::LoggingLevel::Warn)  \
                     << ::atlas::Logging::get_header(::atlas::LoggingLevel::Warn)  \
                     << "[" << __FILE__ << ":" << __LINE__ << " (" << __func__ << ")] " )

#define ATLAS_ERROR ( ::atlas::Logger(::atlas::LoggingLevel::Error) \
                     << ::atlas::Logging::get_header(::atlas::LoggingLevel::Error) \
                     << "[" << __FILE__ << ":" << __LINE__ << " (" << __func__ << ")] " )

#define ATLAS_DEBUG ( ::atlas::Logger(::atlas::LoggingLevel::Debug) \
                     << ::atlas::Logging::get_header(::atlas::LoggingLevel::Debug) \
                     << "[" << __FILE__ << ":" << __LINE__ << " (" << __func__ << ")] " )

#else



#define ATLAS_INFO  (::atlas::make_null_logger())
#define ATLAS_WARN  (::atlas::make_null_logger())
#define ATLAS_ERROR (::atlas::make_null_logger())
#define ATLAS_DEBUG (::atlas::make_null_logger())

#endif

#endif