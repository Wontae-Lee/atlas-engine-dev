#pragma once

#include <chrono>
#include <exception>
#include <iostream>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include <testkit/internal/cudatest-assertion-internal.h>

namespace cudatest {
class Test;
}

namespace cudatest::internal {

class TestFactoryBase {
public:
    virtual ~TestFactoryBase() = default;
    [[nodiscard]] virtual std::unique_ptr<Test> Create() const = 0;
};

template <typename TestType>
class TestFactory final : public TestFactoryBase {
public:
    [[nodiscard]] std::unique_ptr<Test> Create() const override {
        return std::make_unique<TestType>();
    }
};

struct TestInfo {
    std::string suite_name;
    std::string test_name;
    std::unique_ptr<TestFactoryBase> factory;
};

inline bool WildcardMatch(const char* pattern, const char* text) {
    if (*pattern == '\0') {
        return *text == '\0';
    }
    if (*pattern == '*') {
        return WildcardMatch(pattern + 1, text) || (*text != '\0' && WildcardMatch(pattern, text + 1));
    }
    if (*pattern == '?') {
        return *text != '\0' && WildcardMatch(pattern + 1, text + 1);
    }
    return *pattern == *text && WildcardMatch(pattern + 1, text + 1);
}

inline std::vector<std::string> SplitPatterns(const std::string& text, char delimiter) {
    std::vector<std::string> parts;
    std::string current;
    for (const char ch : text) {
        if (ch == delimiter) {
            if (!current.empty()) {
                parts.push_back(current);
                current.clear();
            }
        } else {
            current.push_back(ch);
        }
    }
    if (!current.empty()) {
        parts.push_back(current);
    }
    if (parts.empty()) {
        parts.push_back("*");
    }
    return parts;
}

class Registry {
public:
    static Registry& Instance() {
        static Registry registry;
        return registry;
    }

    template <typename TestType>
    bool Register(const char* suite_name, const char* test_name) {
        tests_.push_back(TestInfo{suite_name, test_name, std::make_unique<TestFactory<TestType>>()});
        return true;
    }

    void SetFilter(std::string filter) { filter_ = std::move(filter); }
    void EnableListTests(bool value) { list_tests_ = value; }

    [[nodiscard]] const std::vector<TestInfo>& tests() const { return tests_; }

    [[nodiscard]] bool ShouldRun(const TestInfo& test_info) const {
        const std::string full_name = test_info.suite_name + "." + test_info.test_name;
        const auto dash = filter_.find('-');
        const std::string positive = dash == std::string::npos ? filter_ : filter_.substr(0, dash);
        const std::string negative = dash == std::string::npos ? "" : filter_.substr(dash + 1);

        bool matched_positive = false;
        for (const auto& pattern : SplitPatterns(positive.empty() ? "*" : positive, ':')) {
            if (WildcardMatch(pattern.c_str(), full_name.c_str())) {
                matched_positive = true;
                break;
            }
        }

        if (!matched_positive) {
            return false;
        }

        for (const auto& pattern : SplitPatterns(negative, ':')) {
            if (!negative.empty() && WildcardMatch(pattern.c_str(), full_name.c_str())) {
                return false;
            }
        }
        return true;
    }

    [[nodiscard]] bool list_tests() const { return list_tests_; }

private:
    std::vector<TestInfo> tests_;
    std::string filter_ = "*";
    bool list_tests_ = false;
};

inline void PrintFailure(const FailureRecord& failure) {
    std::cout << failure.file << ":" << failure.line << ": Failure\n" << failure.message << '\n';
}

inline bool RunSingleTest(const TestInfo& info) {
    auto instance = info.factory->Create();
    TestContext context{info.suite_name, info.test_name};
    g_current_test_context = &context;

    const auto started_at = std::chrono::steady_clock::now();
    std::cout << "[ RUN      ] " << info.suite_name << "." << info.test_name << '\n';

    try {
        instance->SetUp();
        instance->TestBody();
    } catch (const FatalFailureException&) {
    } catch (const std::exception& ex) {
        context.AddFailure(__FILE__, __LINE__, std::string("Unhandled std::exception: ") + ex.what(), true);
    } catch (...) {
        context.AddFailure(__FILE__, __LINE__, "Unhandled unknown exception", true);
    }

    try {
        instance->TearDown();
    } catch (const FatalFailureException&) {
    } catch (const std::exception& ex) {
        context.AddFailure(__FILE__, __LINE__, std::string("Exception in TearDown(): ") + ex.what(), true);
    } catch (...) {
        context.AddFailure(__FILE__, __LINE__, "Unknown exception in TearDown()", true);
    }

    g_current_test_context = nullptr;

    for (const auto& failure : context.failures) {
        PrintFailure(failure);
    }

    const auto elapsed_ms = std::chrono::duration_cast<std::chrono::milliseconds>(
            std::chrono::steady_clock::now() - started_at);

    if (context.failed()) {
        std::cout << "[  FAILED  ] " << info.suite_name << "." << info.test_name << " (" << elapsed_ms.count()
                  << " ms)\n";
        return false;
    }

    std::cout << "[       OK ] " << info.suite_name << "." << info.test_name << " (" << elapsed_ms.count()
              << " ms)\n";
    return true;
}

}  // namespace cudatest::internal
