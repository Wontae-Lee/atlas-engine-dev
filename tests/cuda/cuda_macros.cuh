#pragma once

#include <atlas/core/macros.h>

#include <algorithm>
#include <cmath>
#include <cstring>
#include <exception>
#include <functional>
#include <iostream>
#include <limits>
#include <sstream>
#include <string>
#include <string_view>
#include <type_traits>
#include <utility>
#include <vector>

namespace atlas::test::cuda {

using TestFunction = void (*)();

struct TestCase {
    std::string suite_name;
    std::string test_name;
    TestFunction function = nullptr;
};

struct TestContext {
    bool failed          = false;
    bool abort_requested = false;
    bool skipped         = false;
    std::string skip_message {};
    std::vector<std::string> failures {};
};

class AssertionFailure final : public std::exception {
public:
    explicit AssertionFailure(std::string message) noexcept
        : _message(std::move(message)) { }

    const char*
    what() const noexcept override {
        return _message.c_str();
    }

private:
    std::string _message;
};

class SkipTest final : public std::exception {
public:
    explicit SkipTest(std::string message) noexcept
        : _message(std::move(message)) { }

    const char*
    what() const noexcept override {
        return _message.c_str();
    }

private:
    std::string _message;
};

inline std::vector<TestCase>&
registry() {
    static std::vector<TestCase> tests {};
    return tests;
}

inline TestContext*&
current_context() {
    static TestContext* context = nullptr;
    return context;
}

inline void
register_test(std::string suite_name,
              std::string test_name,
              TestFunction function) {
    registry().push_back(TestCase {
        std::move(suite_name),
        std::move(test_name),
        function });
}

inline void
record_failure(const char* file,
               int line,
               std::string_view expression,
               std::string message,
               bool fatal) {
    auto* context = current_context();
    if (context == nullptr) {
        throw AssertionFailure("CUDA test context is not active.");
    }

    std::ostringstream oss;
    oss << file << ":" << line << ": failure\n"
        << "  expression: " << expression;

    if (!message.empty()) {
        oss << "\n  detail: " << message;
    }

    context->failed = true;
    context->failures.push_back(oss.str());

    if (fatal) {
        context->abort_requested = true;
        throw AssertionFailure(oss.str());
    }
}

inline void
record_skip(const char* file,
            int line,
            std::string message) {
    auto* context = current_context();
    if (context == nullptr) {
        throw SkipTest("CUDA test context is not active.");
    }

    std::ostringstream oss;
    oss << file << ":" << line << ": skipped";
    if (!message.empty()) {
        oss << "\n  detail: " << message;
    }

    context->skipped      = true;
    context->skip_message = oss.str();
    throw SkipTest(oss.str());
}

template <typename L, typename R>
inline void
expect_eq(const L& lhs,
          const R& rhs,
          const char* lhs_expr,
          const char* rhs_expr,
          const char* file,
          int line,
          bool fatal) {
    if (!(lhs == rhs)) {
        std::ostringstream oss;
        oss << lhs_expr << " != " << rhs_expr;
        record_failure(file, line, "EXPECT_EQ", oss.str(), fatal);
    }
}

template <typename L, typename R>
inline void
expect_ne(const L& lhs,
          const R& rhs,
          const char* lhs_expr,
          const char* rhs_expr,
          const char* file,
          int line,
          bool fatal) {
    if (!(lhs != rhs)) {
        std::ostringstream oss;
        oss << lhs_expr << " == " << rhs_expr;
        record_failure(file, line, "EXPECT_NE", oss.str(), fatal);
    }
}

template <typename L, typename R, typename E>
inline void
expect_near(const L& lhs,
            const R& rhs,
            const E& eps,
            const char* lhs_expr,
            const char* rhs_expr,
            const char* eps_expr,
            const char* file,
            int line,
            bool fatal) {
    using std::abs;
    if (!(abs(lhs - rhs) <= eps)) {
        std::ostringstream oss;
        oss << "|" << lhs_expr << " - " << rhs_expr << "| > " << eps_expr;
        record_failure(file, line, "EXPECT_NEAR", oss.str(), fatal);
    }
}

template <typename L, typename R>
inline void
expect_cmp(bool passed,
           const char* opname,
           const L& lhs,
           const R& rhs,
           const char* lhs_expr,
           const char* rhs_expr,
           const char* file,
           int line,
           bool fatal) {
    if (!passed) {
        std::ostringstream oss;
        oss << lhs_expr << " " << opname << " " << rhs_expr
            << " failed (" << lhs << " vs " << rhs << ")";
        record_failure(file, line, opname, oss.str(), fatal);
    }
}

template <typename L, typename R>
inline void
expect_streq(const L& lhs,
             const R& rhs,
             const char* lhs_expr,
             const char* rhs_expr,
             const char* file,
             int line,
             bool fatal) {
    const char* lhs_str = lhs;
    const char* rhs_str = rhs;
    const bool passed   = (lhs_str == nullptr && rhs_str == nullptr)
                       || (lhs_str != nullptr && rhs_str != nullptr && std::strcmp(lhs_str, rhs_str) == 0);

    if (!passed) {
        std::ostringstream oss;
        oss << lhs_expr << " != " << rhs_expr;
        record_failure(file, line, "EXPECT_STREQ", oss.str(), fatal);
    }
}

template <typename L, typename R>
inline void
expect_float_eq(const L& lhs,
                const R& rhs,
                const char* lhs_expr,
                const char* rhs_expr,
                const char* file,
                int line,
                bool fatal) {
    using Common = std::common_type_t<L, R>;
    using std::abs;
    const Common lhs_value = static_cast<Common>(lhs);
    const Common rhs_value = static_cast<Common>(rhs);
    const Common scale     = std::max<Common>({ Common(1), abs(lhs_value), abs(rhs_value) });
    const Common tol       = std::numeric_limits<Common>::epsilon() * Common(4) * scale;

    if (!(abs(lhs_value - rhs_value) <= tol)) {
        std::ostringstream oss;
        oss << lhs_expr << " != " << rhs_expr;
        record_failure(file, line, "EXPECT_FLOAT_EQ", oss.str(), fatal);
    }
}

template <typename Exception, typename Fn>
inline void
expect_throw(Fn&& fn,
             const char* expr,
             const char* exception_expr,
             const char* file,
             int line,
             bool fatal) {
    try {
        std::forward<Fn>(fn)();
    } catch (const Exception&) {
        return;
    } catch (const std::exception& e) {
        std::ostringstream oss;
        oss << expr << " threw unexpected exception instead of " << exception_expr
            << ": " << e.what();
        record_failure(file, line, "EXPECT_THROW", oss.str(), fatal);
        return;
    } catch (...) {
        std::ostringstream oss;
        oss << expr << " threw unexpected non-standard exception instead of " << exception_expr;
        record_failure(file, line, "EXPECT_THROW", oss.str(), fatal);
        return;
    }

    std::ostringstream oss;
    oss << expr << " did not throw " << exception_expr;
    record_failure(file, line, "EXPECT_THROW", oss.str(), fatal);
}

template <typename Fn>
inline void
expect_no_throw(Fn&& fn,
                const char* expr,
                const char* file,
                int line,
                bool fatal) {
    try {
        std::forward<Fn>(fn)();
    } catch (const std::exception& e) {
        std::ostringstream oss;
        oss << expr << " threw unexpected exception: " << e.what();
        record_failure(file, line, "EXPECT_NO_THROW", oss.str(), fatal);
    } catch (...) {
        std::ostringstream oss;
        oss << expr << " threw unexpected non-standard exception";
        record_failure(file, line, "EXPECT_NO_THROW", oss.str(), fatal);
    }
}

inline int
run_all_cuda_tests() {
    int failed_count = 0;
    int skipped_count = 0;

    for (const auto& test : registry()) {
        TestContext context {};
        current_context() = &context;

        std::clog << "[ RUN      ] " << test.suite_name << "." << test.test_name << '\n';

        try {
            test.function();
        } catch (const AssertionFailure&) {
        } catch (const SkipTest&) {
        } catch (const std::exception& e) {
            record_failure(__FILE__, __LINE__, "unexpected exception", e.what(), false);
        } catch (...) {
            record_failure(__FILE__, __LINE__, "unexpected exception", "unknown exception", false);
        }

        if (context.skipped) {
            ++skipped_count;
            std::clog << "[  SKIPPED ] " << test.suite_name << "." << test.test_name << '\n';
            std::clog << context.skip_message << '\n';
        } else if (context.failed) {
            ++failed_count;
            std::clog << "[  FAILED  ] " << test.suite_name << "." << test.test_name << '\n';
            for (const auto& failure : context.failures) {
                std::clog << failure << '\n';
            }
        } else {
            std::clog << "[       OK ] " << test.suite_name << "." << test.test_name << '\n';
        }

        current_context() = nullptr;
    }

    std::clog << "[==========] " << registry().size() << " CUDA tests ran.\n";
    std::clog << "[  PASSED  ] " << (registry().size() - static_cast<std::size_t>(failed_count)) << " tests.\n";
    if (skipped_count > 0) {
        std::clog << "[  SKIPPED ] " << skipped_count << " tests.\n";
    }

    if (failed_count > 0) {
        std::clog << "[  FAILED  ] " << failed_count << " tests.\n";
    }

    return failed_count == 0 ? 0 : 1;
}

template <typename RegistrarTag = void>
struct TestRegistrar final {
    TestRegistrar(const char* suite_name,
                  const char* test_name,
                  TestFunction function) {
        register_test(suite_name, test_name, function);
    }
};

}

#define CUDA_TEST(SuiteName, TestName)                                                                                                     \
    static void SuiteName##_##TestName##_impl();                                                                                           \
    static ::atlas::test::cuda::TestRegistrar<> SuiteName##_##TestName##_registrar(#SuiteName, #TestName, &SuiteName##_##TestName##_impl); \
    static void SuiteName##_##TestName##_impl()

#define CUDA_EXPECT_TRUE(expr)                                                                      \
    do {                                                                                            \
        if (!(expr)) {                                                                              \
            ::atlas::test::cuda::record_failure(__FILE__, __LINE__, #expr, "expected true", false); \
        }                                                                                           \
    } while (false)

#define CUDA_EXPECT_FALSE(expr)                                                                      \
    do {                                                                                             \
        if (expr) {                                                                                  \
            ::atlas::test::cuda::record_failure(__FILE__, __LINE__, #expr, "expected false", false); \
        }                                                                                            \
    } while (false)

#define CUDA_ASSERT_TRUE(expr)                                                                     \
    do {                                                                                           \
        if (!(expr)) {                                                                             \
            ::atlas::test::cuda::record_failure(__FILE__, __LINE__, #expr, "expected true", true); \
        }                                                                                          \
    } while (false)

#define CUDA_ASSERT_FALSE(expr)                                                                     \
    do {                                                                                            \
        if (expr) {                                                                                 \
            ::atlas::test::cuda::record_failure(__FILE__, __LINE__, #expr, "expected false", true); \
        }                                                                                           \
    } while (false)

#define CUDA_EXPECT_EQ(lhs, rhs)                                                             \
    do {                                                                                     \
        ::atlas::test::cuda::expect_eq((lhs), (rhs), #lhs, #rhs, __FILE__, __LINE__, false); \
    } while (false)

#define CUDA_EXPECT_NE(lhs, rhs)                                                             \
    do {                                                                                     \
        ::atlas::test::cuda::expect_ne((lhs), (rhs), #lhs, #rhs, __FILE__, __LINE__, false); \
    } while (false)

#define CUDA_EXPECT_GT(lhs, rhs)                                                                                       \
    do {                                                                                                               \
        ::atlas::test::cuda::expect_cmp(((lhs) > (rhs)), "EXPECT_GT", (lhs), (rhs), #lhs, #rhs, __FILE__, __LINE__, false); \
    } while (false)

#define CUDA_EXPECT_LT(lhs, rhs)                                                                                       \
    do {                                                                                                               \
        ::atlas::test::cuda::expect_cmp(((lhs) < (rhs)), "EXPECT_LT", (lhs), (rhs), #lhs, #rhs, __FILE__, __LINE__, false); \
    } while (false)

#define CUDA_EXPECT_GE(lhs, rhs)                                                                                        \
    do {                                                                                                                \
        ::atlas::test::cuda::expect_cmp(((lhs) >= (rhs)), "EXPECT_GE", (lhs), (rhs), #lhs, #rhs, __FILE__, __LINE__, false); \
    } while (false)

#define CUDA_EXPECT_LE(lhs, rhs)                                                                                        \
    do {                                                                                                                \
        ::atlas::test::cuda::expect_cmp(((lhs) <= (rhs)), "EXPECT_LE", (lhs), (rhs), #lhs, #rhs, __FILE__, __LINE__, false); \
    } while (false)

#define CUDA_EXPECT_NEAR(lhs, rhs, eps)                                                                     \
    do {                                                                                                    \
        ::atlas::test::cuda::expect_near((lhs), (rhs), (eps), #lhs, #rhs, #eps, __FILE__, __LINE__, false); \
    } while (false)

#define CUDA_EXPECT_FLOAT_EQ(lhs, rhs)                                                                      \
    do {                                                                                                    \
        ::atlas::test::cuda::expect_float_eq((lhs), (rhs), #lhs, #rhs, __FILE__, __LINE__, false);        \
    } while (false)

#define CUDA_EXPECT_DOUBLE_EQ(lhs, rhs) CUDA_EXPECT_FLOAT_EQ((lhs), (rhs))

#define CUDA_EXPECT_STREQ(lhs, rhs)                                                                \
    do {                                                                                            \
        ::atlas::test::cuda::expect_streq((lhs), (rhs), #lhs, #rhs, __FILE__, __LINE__, false);   \
    } while (false)

#define CUDA_EXPECT_THROW(expr, ExceptionType)                                                                          \
    do {                                                                                                                \
        ::atlas::test::cuda::expect_throw<ExceptionType>([&]() { (void)(expr); }, #expr, #ExceptionType, __FILE__, __LINE__, false); \
    } while (false)

#define CUDA_EXPECT_NO_THROW(expr)                                                        \
    do {                                                                                   \
        ::atlas::test::cuda::expect_no_throw([&]() { (void)(expr); }, #expr, __FILE__, __LINE__, false); \
    } while (false)

#define CUDA_ASSERT_EQ(lhs, rhs)                                                            \
    do {                                                                                    \
        ::atlas::test::cuda::expect_eq((lhs), (rhs), #lhs, #rhs, __FILE__, __LINE__, true); \
    } while (false)

#define CUDA_ASSERT_NE(lhs, rhs)                                                            \
    do {                                                                                    \
        ::atlas::test::cuda::expect_ne((lhs), (rhs), #lhs, #rhs, __FILE__, __LINE__, true); \
    } while (false)

#define CUDA_ASSERT_GT(lhs, rhs)                                                                                      \
    do {                                                                                                              \
        ::atlas::test::cuda::expect_cmp(((lhs) > (rhs)), "ASSERT_GT", (lhs), (rhs), #lhs, #rhs, __FILE__, __LINE__, true); \
    } while (false)

#define CUDA_ASSERT_LT(lhs, rhs)                                                                                      \
    do {                                                                                                              \
        ::atlas::test::cuda::expect_cmp(((lhs) < (rhs)), "ASSERT_LT", (lhs), (rhs), #lhs, #rhs, __FILE__, __LINE__, true); \
    } while (false)

#define CUDA_ASSERT_GE(lhs, rhs)                                                                                       \
    do {                                                                                                               \
        ::atlas::test::cuda::expect_cmp(((lhs) >= (rhs)), "ASSERT_GE", (lhs), (rhs), #lhs, #rhs, __FILE__, __LINE__, true); \
    } while (false)

#define CUDA_ASSERT_LE(lhs, rhs)                                                                                       \
    do {                                                                                                               \
        ::atlas::test::cuda::expect_cmp(((lhs) <= (rhs)), "ASSERT_LE", (lhs), (rhs), #lhs, #rhs, __FILE__, __LINE__, true); \
    } while (false)

#define CUDA_ASSERT_NEAR(lhs, rhs, eps)                                                                    \
    do {                                                                                                   \
        ::atlas::test::cuda::expect_near((lhs), (rhs), (eps), #lhs, #rhs, #eps, __FILE__, __LINE__, true); \
    } while (false)

#define CUDA_ASSERT_FLOAT_EQ(lhs, rhs)                                                                     \
    do {                                                                                                   \
        ::atlas::test::cuda::expect_float_eq((lhs), (rhs), #lhs, #rhs, __FILE__, __LINE__, true);        \
    } while (false)

#define CUDA_ASSERT_DOUBLE_EQ(lhs, rhs) CUDA_ASSERT_FLOAT_EQ((lhs), (rhs))

#define CUDA_ASSERT_STREQ(lhs, rhs)                                                               \
    do {                                                                                           \
        ::atlas::test::cuda::expect_streq((lhs), (rhs), #lhs, #rhs, __FILE__, __LINE__, true);   \
    } while (false)

#define CUDA_ASSERT_THROW(expr, ExceptionType)                                                                           \
    do {                                                                                                                 \
        ::atlas::test::cuda::expect_throw<ExceptionType>([&]() { (void)(expr); }, #expr, #ExceptionType, __FILE__, __LINE__, true); \
    } while (false)

#define CUDA_ASSERT_NO_THROW(expr)                                                         \
    do {                                                                                    \
        ::atlas::test::cuda::expect_no_throw([&]() { (void)(expr); }, #expr, __FILE__, __LINE__, true); \
    } while (false)

#define CUDA_SKIP(message) ::atlas::test::cuda::record_skip(__FILE__, __LINE__, (message))

#define CUDA_SUCCEED() do { } while (false)

#define CUDA_RUN_ALL_TESTS() ::atlas::test::cuda::run_all_cuda_tests()
