#pragma once

#include <algorithm>
#include <functional>
#include <iostream>
#include <string>
#include <vector>

namespace cudatest {

class Test {
public:
    virtual ~Test() = default;
    virtual void SetUp() {}
    virtual void TearDown() {}
    virtual void TestBody() = 0;
};

}  // namespace cudatest

#include <testkit/internal/cudatest-registry.h>

namespace cudatest {

inline void InitCudaTest(int* argc, char** argv) {
    if (argc == nullptr || argv == nullptr) {
        return;
    }

    int write_index = 1;
    for (int read_index = 1; read_index < *argc; ++read_index) {
        const std::string argument = argv[read_index];
        if (argument.rfind("--gtest_filter=", 0) == 0) {
            internal::Registry::Instance().SetFilter(argument.substr(std::string("--gtest_filter=").size()));
            continue;
        }
        if (argument == "--gtest_list_tests") {
            internal::Registry::Instance().EnableListTests(true);
            continue;
        }
        argv[write_index++] = argv[read_index];
    }
    *argc = write_index;
}

inline void InitGoogleTest(int* argc, char** argv) {
    InitCudaTest(argc, argv);
}

inline int RUN_ALL_TESTS() {
    auto& registry = internal::Registry::Instance();
    if (registry.list_tests()) {
        std::string current_suite;
        for (const auto& test : registry.tests()) {
            if (!registry.ShouldRun(test)) {
                continue;
            }
            if (current_suite != test.suite_name) {
                current_suite = test.suite_name;
                std::cout << current_suite << ".\n";
            }
            std::cout << "  " << test.test_name << '\n';
        }
        return 0;
    }

    std::vector<std::reference_wrapper<const internal::TestInfo>> selected;
    for (const auto& test : registry.tests()) {
        if (registry.ShouldRun(test)) {
            selected.push_back(std::cref(test));
        }
    }

    std::vector<std::string> suites;
    for (const auto& test_ref : selected) {
        const auto& suite_name = test_ref.get().suite_name;
        if (std::find(suites.begin(), suites.end(), suite_name) == suites.end()) {
            suites.push_back(suite_name);
        }
    }

    std::cout << "[==========] Running " << selected.size() << " tests from " << suites.size() << " test suites.\n";

    std::vector<std::string> failed_tests;
    for (const auto& suite : suites) {
        size_t suite_count = 0;
        for (const auto& test_ref : selected) {
            if (test_ref.get().suite_name == suite) {
                ++suite_count;
            }
        }

        std::cout << "[----------] " << suite_count << " tests from " << suite << '\n';
        for (const auto& test_ref : selected) {
            if (test_ref.get().suite_name != suite) {
                continue;
            }
            if (!internal::RunSingleTest(test_ref.get())) {
                failed_tests.push_back(test_ref.get().suite_name + "." + test_ref.get().test_name);
            }
        }
    }

    std::cout << "[==========] " << selected.size() << " tests from " << suites.size() << " test suites ran.\n";
    std::cout << "[  PASSED  ] " << (selected.size() - failed_tests.size()) << " tests.\n";

    if (!failed_tests.empty()) {
        std::cout << "[  FAILED  ] " << failed_tests.size() << " tests, listed below:\n";
        for (const auto& failed_test : failed_tests) {
            std::cout << "[  FAILED  ] " << failed_test << '\n';
        }
    }

    return failed_tests.empty() ? 0 : 1;
}

}  // namespace cudatest

namespace testing = ::cudatest;
