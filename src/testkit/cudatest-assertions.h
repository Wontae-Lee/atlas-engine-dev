#pragma once

#include <testkit/cudatest-test.h>
#include <testkit/internal/cudatest-assertion-internal.h>

#define CUDATEST_INTERNAL_TEST_CLASS_NAME_(test_suite_name, test_name) test_suite_name##_##test_name##_Test
#define CUDATEST_INTERNAL_TEST_REGISTRAR_NAME_(test_suite_name, test_name) \
    test_suite_name##_##test_name##_registrar_

#define TEST(test_suite_name, test_name)                                                                  \
    class CUDATEST_INTERNAL_TEST_CLASS_NAME_(test_suite_name, test_name) : public ::cudatest::Test {     \
    public:                                                                                               \
        void TestBody() override;                                                                         \
    };                                                                                                    \
    static const bool CUDATEST_INTERNAL_TEST_REGISTRAR_NAME_(test_suite_name, test_name) =               \
            ::cudatest::internal::Registry::Instance().Register<                                          \
                    CUDATEST_INTERNAL_TEST_CLASS_NAME_(test_suite_name, test_name)>(                      \
                    #test_suite_name, #test_name);                                                        \
    void CUDATEST_INTERNAL_TEST_CLASS_NAME_(test_suite_name, test_name)::TestBody()

#define TEST_F(test_fixture, test_name)                                                                   \
    class CUDATEST_INTERNAL_TEST_CLASS_NAME_(test_fixture, test_name) : public test_fixture {            \
    public:                                                                                               \
        void TestBody() override;                                                                         \
    };                                                                                                    \
    static const bool CUDATEST_INTERNAL_TEST_REGISTRAR_NAME_(test_fixture, test_name) =                  \
            ::cudatest::internal::Registry::Instance().Register<                                          \
                    CUDATEST_INTERNAL_TEST_CLASS_NAME_(test_fixture, test_name)>(                         \
                    #test_fixture, #test_name);                                                           \
    void CUDATEST_INTERNAL_TEST_CLASS_NAME_(test_fixture, test_name)::TestBody()

#define EXPECT_TRUE(condition)                                                                            \
    ::cudatest::internal::MakeBooleanAssertion(__FILE__, __LINE__, false, static_cast<bool>(condition),  \
                                               true, #condition, "true")
#define ASSERT_TRUE(condition)                                                                            \
    ::cudatest::internal::MakeBooleanAssertion(__FILE__, __LINE__, true, static_cast<bool>(condition),   \
                                               true, #condition, "true")
#define EXPECT_FALSE(condition)                                                                           \
    ::cudatest::internal::MakeBooleanAssertion(__FILE__, __LINE__, false, static_cast<bool>(condition),  \
                                               false, #condition, "false")
#define ASSERT_FALSE(condition)                                                                           \
    ::cudatest::internal::MakeBooleanAssertion(__FILE__, __LINE__, true, static_cast<bool>(condition),   \
                                               false, #condition, "false")

#define EXPECT_EQ(val1, val2)                                                                             \
    ::cudatest::internal::MakeBinaryAssertion(__FILE__, __LINE__, false, #val1, #val2, "==", (val1),    \
                                              (val2), [](const auto& lhs, const auto& rhs) {              \
                                                  return lhs == rhs;                                      \
                                              })
#define ASSERT_EQ(val1, val2)                                                                             \
    ::cudatest::internal::MakeBinaryAssertion(__FILE__, __LINE__, true, #val1, #val2, "==", (val1),     \
                                              (val2), [](const auto& lhs, const auto& rhs) {              \
                                                  return lhs == rhs;                                      \
                                              })
#define EXPECT_NE(val1, val2)                                                                             \
    ::cudatest::internal::MakeBinaryAssertion(__FILE__, __LINE__, false, #val1, #val2, "!=", (val1),    \
                                              (val2), [](const auto& lhs, const auto& rhs) {              \
                                                  return lhs != rhs;                                      \
                                              })
#define ASSERT_NE(val1, val2)                                                                             \
    ::cudatest::internal::MakeBinaryAssertion(__FILE__, __LINE__, true, #val1, #val2, "!=", (val1),     \
                                              (val2), [](const auto& lhs, const auto& rhs) {              \
                                                  return lhs != rhs;                                      \
                                              })
#define EXPECT_LT(val1, val2)                                                                             \
    ::cudatest::internal::MakeBinaryAssertion(__FILE__, __LINE__, false, #val1, #val2, "<", (val1),     \
                                              (val2), [](const auto& lhs, const auto& rhs) {              \
                                                  return lhs < rhs;                                       \
                                              })
#define ASSERT_LT(val1, val2)                                                                             \
    ::cudatest::internal::MakeBinaryAssertion(__FILE__, __LINE__, true, #val1, #val2, "<", (val1),      \
                                              (val2), [](const auto& lhs, const auto& rhs) {              \
                                                  return lhs < rhs;                                       \
                                              })
#define EXPECT_LE(val1, val2)                                                                             \
    ::cudatest::internal::MakeBinaryAssertion(__FILE__, __LINE__, false, #val1, #val2, "<=", (val1),    \
                                              (val2), [](const auto& lhs, const auto& rhs) {              \
                                                  return lhs <= rhs;                                      \
                                              })
#define ASSERT_LE(val1, val2)                                                                             \
    ::cudatest::internal::MakeBinaryAssertion(__FILE__, __LINE__, true, #val1, #val2, "<=", (val1),     \
                                              (val2), [](const auto& lhs, const auto& rhs) {              \
                                                  return lhs <= rhs;                                      \
                                              })
#define EXPECT_GT(val1, val2)                                                                             \
    ::cudatest::internal::MakeBinaryAssertion(__FILE__, __LINE__, false, #val1, #val2, ">", (val1),     \
                                              (val2), [](const auto& lhs, const auto& rhs) {              \
                                                  return lhs > rhs;                                       \
                                              })
#define ASSERT_GT(val1, val2)                                                                             \
    ::cudatest::internal::MakeBinaryAssertion(__FILE__, __LINE__, true, #val1, #val2, ">", (val1),      \
                                              (val2), [](const auto& lhs, const auto& rhs) {              \
                                                  return lhs > rhs;                                       \
                                              })
#define EXPECT_GE(val1, val2)                                                                             \
    ::cudatest::internal::MakeBinaryAssertion(__FILE__, __LINE__, false, #val1, #val2, ">=", (val1),    \
                                              (val2), [](const auto& lhs, const auto& rhs) {              \
                                                  return lhs >= rhs;                                      \
                                              })
#define ASSERT_GE(val1, val2)                                                                             \
    ::cudatest::internal::MakeBinaryAssertion(__FILE__, __LINE__, true, #val1, #val2, ">=", (val1),     \
                                              (val2), [](const auto& lhs, const auto& rhs) {              \
                                                  return lhs >= rhs;                                      \
                                              })

#define EXPECT_STREQ(str1, str2)                                                                          \
    ::cudatest::internal::MakeCStringAssertion(__FILE__, __LINE__, false, #str1, #str2, (str1), (str2), \
                                               true, false)
#define ASSERT_STREQ(str1, str2)                                                                          \
    ::cudatest::internal::MakeCStringAssertion(__FILE__, __LINE__, true, #str1, #str2, (str1), (str2),  \
                                               true, false)
#define EXPECT_STRNE(str1, str2)                                                                          \
    ::cudatest::internal::MakeCStringAssertion(__FILE__, __LINE__, false, #str1, #str2, (str1), (str2), \
                                               false, false)
#define ASSERT_STRNE(str1, str2)                                                                          \
    ::cudatest::internal::MakeCStringAssertion(__FILE__, __LINE__, true, #str1, #str2, (str1), (str2),  \
                                               false, false)
#define EXPECT_STRCASEEQ(str1, str2)                                                                      \
    ::cudatest::internal::MakeCStringAssertion(__FILE__, __LINE__, false, #str1, #str2, (str1), (str2), \
                                               true, true)
#define ASSERT_STRCASEEQ(str1, str2)                                                                      \
    ::cudatest::internal::MakeCStringAssertion(__FILE__, __LINE__, true, #str1, #str2, (str1), (str2),  \
                                               true, true)
#define EXPECT_STRCASENE(str1, str2)                                                                      \
    ::cudatest::internal::MakeCStringAssertion(__FILE__, __LINE__, false, #str1, #str2, (str1), (str2), \
                                               false, true)
#define ASSERT_STRCASENE(str1, str2)                                                                      \
    ::cudatest::internal::MakeCStringAssertion(__FILE__, __LINE__, true, #str1, #str2, (str1), (str2),  \
                                               false, true)

#define EXPECT_NEAR(val1, val2, abs_error)                                                                \
    ::cudatest::internal::MakeNearAssertion(__FILE__, __LINE__, false, #val1, #val2, #abs_error,        \
                                            (val1), (val2), (abs_error))
#define ASSERT_NEAR(val1, val2, abs_error)                                                                \
    ::cudatest::internal::MakeNearAssertion(__FILE__, __LINE__, true, #val1, #val2, #abs_error,         \
                                            (val1), (val2), (abs_error))
#define EXPECT_FLOAT_EQ(val1, val2)                                                                       \
    ::cudatest::internal::MakeFloatingEqAssertion<float>(__FILE__, __LINE__, false, #val1, #val2,       \
                                                         static_cast<float>(val1), static_cast<float>(val2))
#define ASSERT_FLOAT_EQ(val1, val2)                                                                       \
    ::cudatest::internal::MakeFloatingEqAssertion<float>(__FILE__, __LINE__, true, #val1, #val2,        \
                                                         static_cast<float>(val1), static_cast<float>(val2))
#define EXPECT_DOUBLE_EQ(val1, val2)                                                                      \
    ::cudatest::internal::MakeFloatingEqAssertion<double>(__FILE__, __LINE__, false, #val1, #val2,      \
                                                          static_cast<double>(val1), static_cast<double>(val2))
#define ASSERT_DOUBLE_EQ(val1, val2)                                                                      \
    ::cudatest::internal::MakeFloatingEqAssertion<double>(__FILE__, __LINE__, true, #val1, #val2,       \
                                                          static_cast<double>(val1), static_cast<double>(val2))

#define EXPECT_THROW(statement, exception_type)                                                           \
    ::cudatest::internal::MakeThrowAssertion<exception_type>(                                             \
            __FILE__, __LINE__, false, #statement, #exception_type, [&]() { statement; })
#define ASSERT_THROW(statement, exception_type)                                                           \
    ::cudatest::internal::MakeThrowAssertion<exception_type>(                                             \
            __FILE__, __LINE__, true, #statement, #exception_type, [&]() { statement; })
#define EXPECT_NO_THROW(statement)                                                                        \
    ::cudatest::internal::MakeNoThrowAssertion(__FILE__, __LINE__, false, #statement, [&]() { statement; })
#define ASSERT_NO_THROW(statement)                                                                        \
    ::cudatest::internal::MakeNoThrowAssertion(__FILE__, __LINE__, true, #statement, [&]() { statement; })
#define EXPECT_ANY_THROW(statement)                                                                       \
    ::cudatest::internal::MakeAnyThrowAssertion(__FILE__, __LINE__, false, #statement, [&]() { statement; })
#define ASSERT_ANY_THROW(statement)                                                                       \
    ::cudatest::internal::MakeAnyThrowAssertion(__FILE__, __LINE__, true, #statement, [&]() { statement; })

#define SUCCEED() ::cudatest::internal::Assertion(true, false, __FILE__, __LINE__, "Succeeded")
#define ADD_FAILURE() ::cudatest::internal::Assertion(false, false, __FILE__, __LINE__, "Failed")
#define FAIL() ::cudatest::internal::Assertion(false, true, __FILE__, __LINE__, "Failed")
