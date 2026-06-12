#pragma once

#include <cstddef>
#include <type_traits>
#include <utility>

namespace atlas::detail {

template <typename E>
using expr_value_t = std::decay_t<decltype(std::declval<const E&>()[std::size_t { 0 }])>;

template <typename T>
struct Add {

    ATLAS_ALL_DEVICE T
    operator()(T a, T b) const noexcept { return a + b; }
};

template <typename T>
struct Sub {

    ATLAS_ALL_DEVICE T
    operator()(T a, T b) const noexcept { return a - b; }
};

template <typename T>
struct RSub {

    ATLAS_ALL_DEVICE T
    operator()(T a, T b) const noexcept { return b - a; }
};

template <typename T>
struct Mul {

    ATLAS_ALL_DEVICE T
    operator()(T a, T b) const noexcept { return a * b; }
};

template <typename T>
struct Div {

    ATLAS_ALL_DEVICE T
    operator()(T a, T b) const noexcept { return a / b; }
};

template <typename T>
struct RDiv {

    ATLAS_ALL_DEVICE T
    operator()(T a, T b) const noexcept { return b / a; }
};

template <typename T>
struct Less {
    ATLAS_ALL_DEVICE bool
    operator()(T a, T b) const noexcept { return a < b; }
};

template <typename T>
struct LessEqual {
    ATLAS_ALL_DEVICE bool
    operator()(T a, T b) const noexcept { return a <= b; }
};

template <typename T>
struct Greater {
    ATLAS_ALL_DEVICE bool
    operator()(T a, T b) const noexcept { return a > b; }
};

template <typename T>
struct GreaterEqual {
    ATLAS_ALL_DEVICE bool
    operator()(T a, T b) const noexcept { return a >= b; }
};

template <typename T>
struct Equal {
    ATLAS_ALL_DEVICE bool
    operator()(T a, T b) const noexcept { return a == b; }
};

template <typename T>
struct NotEqual {
    ATLAS_ALL_DEVICE bool
    operator()(T a, T b) const noexcept { return a != b; }
};

template <typename T>
struct Negate {

    ATLAS_ALL_DEVICE T
    operator()(T v) const noexcept { return -v; }
};

template <typename T>
struct Abs {

    ATLAS_ALL_DEVICE T
    operator()(T v) const noexcept { return (v < T(0)) ? -v : v; }
};

template <typename T>
struct Sign {

    ATLAS_ALL_DEVICE T
    operator()(T v) const noexcept { return (T(0) < v) - (v < T(0)); }
};

template <typename From, typename To>
struct TypeCast {

    ATLAS_ALL_DEVICE To
    operator()(const From& v) const noexcept { return static_cast<To>(v); }
};

template <typename T>
struct CompareMin {
    ATLAS_ALL_DEVICE T
    operator()(T a, T b) const noexcept { return (b < a) ? b : a; }
};

template <typename T>
struct CompareMax {
    ATLAS_ALL_DEVICE T
    operator()(T a, T b) const noexcept { return (a < b) ? b : a; }
};

template <typename T>
struct ClampScalar {
    T lo, hi;

    ATLAS_ALL_DEVICE T
    operator()(T v) const noexcept { return (v < lo) ? lo : ((hi < v) ? hi : v); }
};

template <typename T>
struct LogicalAnd {

    ATLAS_ALL_DEVICE bool
    operator()(bool l, bool r) const noexcept {
        (void)sizeof(T);
        return l && r;
    }
};

template <typename T>
struct LogicalOr {

    ATLAS_ALL_DEVICE bool
    operator()(bool l, bool r) const noexcept {
        (void)sizeof(T);
        return l || r;
    }
};

template <typename T>
struct RLess {
    ATLAS_ALL_DEVICE bool
    operator()(T a, T b) const noexcept { return b < a; }
};

template <typename T>
struct RLessEqual {
    ATLAS_ALL_DEVICE bool
    operator()(T a, T b) const noexcept { return b <= a; }
};

template <typename T>
struct RGreater {
    ATLAS_ALL_DEVICE bool
    operator()(T a, T b) const noexcept { return b > a; }
};

template <typename T>
struct RGreaterEqual {
    ATLAS_ALL_DEVICE bool
    operator()(T a, T b) const noexcept { return b >= a; }
};

template <typename T>
struct REqual {
    ATLAS_ALL_DEVICE bool
    operator()(T a, T b) const noexcept { return b == a; }
};

template <typename T>
struct RNotEqual {
    ATLAS_ALL_DEVICE bool
    operator()(T a, T b) const noexcept { return b != a; }
};

}