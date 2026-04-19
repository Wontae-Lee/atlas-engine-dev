#pragma once

#include <iostream>
#include <sstream>
#include <string>
#include <type_traits>
#include <utility>

namespace cudatest::internal {

template <typename T, typename = void>
struct IsStreamable : std::false_type {};

template <typename T>
struct IsStreamable<T, std::void_t<decltype(std::declval<std::ostream&>() << std::declval<const T&>())>>
        : std::true_type {};

template <typename T>
std::string FormatValue(const T& value) {
    std::ostringstream stream;
    stream << std::boolalpha;
    if constexpr (IsStreamable<T>::value) {
        stream << value;
    } else {
        stream << "<non-streamable>";
    }
    return stream.str();
}

inline std::string FormatCString(const char* value) {
    if (value == nullptr) {
        return "nullptr";
    }
    std::ostringstream stream;
    stream << '"' << value << '"';
    return stream.str();
}

}  // namespace cudatest::internal
