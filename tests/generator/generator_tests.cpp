#include "../utilities/test_utils.h"

#include <atlas/generator/generate_operator.h>
#include <atlas/generator/generator.h>

#include <testkit/testkit.h>

namespace {

using atlas::GenerateType;
using atlas::GeneratorHostPtr;
using atlas::Vector3F;
using atlas::GenerateOperator;
using atlas::Generator;
using atlas::UniformGenerateOperator;
using atlas::make_host_shared;
using atlas::test::vec_near;
using atlas::tol;

class DummyGenerator final : public Generator<float> {
public:
    explicit DummyGenerator(Vector3F sample = Vector3F(1, 2, 3))
        : _sample(sample)
        , _operator(UniformGenerateOperator<float>(17u)) {}

    Vector3F
    generate() const override {
        return _sample;
    }

    const GenerateOperator<float>&
    generate_operator() const noexcept override {
        return _operator;
    }

    GenerateOperator<float>
    make_generate_operator() const noexcept override {
        return _operator;
    }

    float
    param0() const noexcept override {
        return 4.0f;
    }

    float
    param1() const noexcept override {
        return 9.0f;
    }

    GenerateType
    type() const noexcept override {
        return GenerateType::uniform;
    }

private:
    Vector3F _sample;
    GenerateOperator<float> _operator;
};

} // namespace

TEST(Generator, DerivedImplementationSatisfiesInterface) {
    // Arrange: create a concrete test generator.
    const DummyGenerator generator;

    // Assert: the implementation satisfies the generator interface contract.
    EXPECT_TRUE(vec_near(generator.generate(), Vector3F(1, 2, 3), tol));
    EXPECT_EQ(generator.type(), GenerateType::uniform);
    EXPECT_NEAR(generator.param0(), 4.0f, tol);
    EXPECT_NEAR(generator.param1(), 9.0f, tol);
    EXPECT_EQ(generator.generate_operator().type, GenerateType::uniform);
    EXPECT_EQ(generator.make_generate_operator().type, GenerateType::uniform);
}

TEST(Generator, HostSharedAliasCanOwnDerivedImplementation) {
    // Act: store a derived generator through the public host pointer alias.
    GeneratorHostPtr<float> generator = make_host_shared<DummyGenerator>(Vector3F(3, 2, 1));

    // Assert: shared ownership preserves the dynamic implementation.
    ASSERT_NE(generator, nullptr);
    EXPECT_TRUE(vec_near(generator->generate(), Vector3F(3, 2, 1), tol));
}
