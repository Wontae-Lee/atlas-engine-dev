#pragma once

#include <algorithm>
#include <cctype>
#include <cmath>
#include <cstring>
#include <exception>
#include <iostream>
#include <limits>
#include <sstream>
#include <string>
#include <utility>
#include <vector>

#include <testkit/internal/cudatest-string.h>

namespace cudatest::internal {

struct FailureRecord {
    std::string file;
    int line = 0;
    std::string message;
    bool fatal = false;
};

struct TestContext {
    std::string suite_name;
    std::string test_name;
    std::vector<FailureRecord> failures;
    int assertion_count = 0;

    [[nodiscard]] bool failed() const { return !failures.empty(); }

    void AddFailure(const char* file, int line, std::string message, bool fatal_failure) {
        failures.push_back(FailureRecord{file, line, std::move(message), fatal_failure});
    }
};

struct FatalFailureException final : std::exception {
    [[nodiscard]] const char* what() const noexcept override { return "cudatest fatal assertion"; }
};

inline thread_local TestContext* g_current_test_context = nullptr;

class Assertion {
public:
    Assertion(bool passed, bool fatal, const char* file, int line, std::string message)
            : passed_(passed), fatal_(fatal), file_(file), line_(line), message_(std::move(message)) {
        if (g_current_test_context != nullptr) {
            ++g_current_test_context->assertion_count;
        }
    }

    Assertion(const Assertion&) = delete;
    Assertion& operator=(const Assertion&) = delete;

    Assertion(Assertion&& other) noexcept
            : passed_(other.passed_),
              fatal_(other.fatal_),
              file_(other.file_),
              line_(other.line_),
              message_(std::move(other.message_)),
              detail_stream_(std::move(other.detail_stream_)),
              consumed_(other.consumed_) {
        other.consumed_ = true;
    }

    ~Assertion() noexcept(false) {
        if (consumed_ || passed_) {
            return;
        }

        std::string full_message = message_;
        const std::string extra = detail_stream_.str();
        if (!extra.empty()) {
            full_message += "\n";
            full_message += extra;
        }

        if (g_current_test_context != nullptr) {
            g_current_test_context->AddFailure(file_, line_, full_message, fatal_);
        } else {
            std::cerr << file_ << ":" << line_ << ": Failure\n" << full_message << '\n';
        }

        if (fatal_) {
            throw FatalFailureException{};
        }
    }

    template <typename T>
    Assertion& operator<<(const T& value) {
        if (!passed_) {
            detail_stream_ << value;
        }
        return *this;
    }

    using StreamManipulator = std::ostream& (*)(std::ostream&);

    Assertion& operator<<(StreamManipulator manipulator) {
        if (!passed_) {
            detail_stream_ << manipulator;
        }
        return *this;
    }

private:
    bool passed_;
    bool fatal_;
    const char* file_;
    int line_;
    std::string message_;
    std::ostringstream detail_stream_;
    bool consumed_ = false;
};

template <typename Predicate, typename Left, typename Right>
Assertion MakeBinaryAssertion(const char* file,
                              int line,
                              bool fatal,
                              const char* lhs_text,
                              const char* rhs_text,
                              const char* op_text,
                              Left&& lhs,
                              Right&& rhs,
                              Predicate&& predicate) {
    const bool passed = predicate(lhs, rhs);
    std::ostringstream message;
    message << "Expected: (" << lhs_text << ") " << op_text << " (" << rhs_text << ")\n"
            << "  Actual: " << FormatValue(lhs) << " vs " << FormatValue(rhs);
    return Assertion(passed, fatal, file, line, message.str());
}

inline Assertion MakeBooleanAssertion(const char* file,
                                      int line,
                                      bool fatal,
                                      bool actual,
                                      bool expected,
                                      const char* expression,
                                      const char* expected_text) {
    const bool passed = actual == expected;
    std::ostringstream message;
    message << "Value of: " << expression << '\n'
            << "  Actual: " << std::boolalpha << actual << '\n'
            << "Expected: " << expected_text;
    return Assertion(passed, fatal, file, line, message.str());
}

inline Assertion MakeCStringAssertion(const char* file,
                                      int line,
                                      bool fatal,
                                      const char* lhs_text,
                                      const char* rhs_text,
                                      const char* lhs,
                                      const char* rhs,
                                      bool expect_equal,
                                      bool ignore_case) {
    const bool both_null = lhs == nullptr && rhs == nullptr;
    bool passed = both_null;

    if (!both_null && lhs != nullptr && rhs != nullptr) {
        if (ignore_case) {
            std::string left(lhs);
            std::string right(rhs);
            for (char& ch : left) {
                ch = static_cast<char>(std::tolower(static_cast<unsigned char>(ch)));
            }
            for (char& ch : right) {
                ch = static_cast<char>(std::tolower(static_cast<unsigned char>(ch)));
            }
            passed = left == right;
        } else {
            passed = std::strcmp(lhs, rhs) == 0;
        }
    }

    if (!expect_equal) {
        passed = !passed;
    }

    std::ostringstream message;
    message << "Expected: " << lhs_text << (expect_equal ? " == " : " != ") << rhs_text << '\n'
            << "  Actual: " << FormatCString(lhs) << " vs " << FormatCString(rhs);
    return Assertion(passed, fatal, file, line, message.str());
}

template <typename Left, typename Right, typename AbsError>
Assertion MakeNearAssertion(const char* file,
                            int line,
                            bool fatal,
                            const char* lhs_text,
                            const char* rhs_text,
                            const char* abs_error_text,
                            Left&& lhs,
                            Right&& rhs,
                            AbsError&& abs_error) {
    const auto diff = std::fabs(static_cast<double>(lhs) - static_cast<double>(rhs));
    const auto tolerance = static_cast<double>(abs_error);
    const bool passed = diff <= tolerance;

    std::ostringstream message;
    message << "The difference between " << lhs_text << " and " << rhs_text << " is " << diff
            << ", which exceeds " << abs_error_text << ", where\n"
            << lhs_text << " evaluates to " << FormatValue(lhs) << ",\n"
            << rhs_text << " evaluates to " << FormatValue(rhs) << ", and\n"
            << abs_error_text << " evaluates to " << FormatValue(abs_error);
    return Assertion(passed, fatal, file, line, message.str());
}

template <typename FloatT>
Assertion MakeFloatingEqAssertion(const char* file,
                                  int line,
                                  bool fatal,
                                  const char* lhs_text,
                                  const char* rhs_text,
                                  FloatT lhs,
                                  FloatT rhs) {
    const auto scale = std::max<FloatT>({FloatT{1}, std::fabs(lhs), std::fabs(rhs)});
    const auto tolerance = std::numeric_limits<FloatT>::epsilon() * FloatT{4} * scale;
    return MakeNearAssertion(file, line, fatal, lhs_text, rhs_text, "floating_tolerance", lhs, rhs, tolerance);
}

template <typename Exception, typename Callable>
Assertion MakeThrowAssertion(const char* file,
                             int line,
                             bool fatal,
                             const char* statement_text,
                             const char* exception_text,
                             Callable&& callable) {
    bool caught_expected = false;
    bool caught_other = false;

    try {
        callable();
    } catch (const Exception&) {
        caught_expected = true;
    } catch (...) {
        caught_other = true;
    }

    std::ostringstream message;
    message << "Expected: " << statement_text << " throws " << exception_text;
    if (caught_other) {
        message << "\n  Actual: it throws a different exception type";
    } else if (!caught_expected) {
        message << "\n  Actual: it throws nothing";
    }
    return Assertion(caught_expected, fatal, file, line, message.str());
}

template <typename Callable>
Assertion MakeNoThrowAssertion(const char* file, int line, bool fatal, const char* statement_text, Callable&& callable) {
    bool threw = false;
    try {
        callable();
    } catch (...) {
        threw = true;
    }

    std::ostringstream message;
    message << "Expected: " << statement_text << " doesn't throw an exception";
    return Assertion(!threw, fatal, file, line, message.str());
}

template <typename Callable>
Assertion MakeAnyThrowAssertion(const char* file, int line, bool fatal, const char* statement_text, Callable&& callable) {
    bool threw = false;
    try {
        callable();
    } catch (...) {
        threw = true;
    }

    std::ostringstream message;
    message << "Expected: " << statement_text << " throws an exception";
    return Assertion(threw, fatal, file, line, message.str());
}

}  // namespace cudatest::internal
