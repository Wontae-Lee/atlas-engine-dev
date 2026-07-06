#include <atlas/generator/generator.h>

#include <atlas/generator/generate.h>

#include <gtest/gtest.h>

namespace {

using atlas::Generate;
using atlas::GenerateType;
using atlas::Generator;
using atlas::GeneratorHostPtr;
using atlas::make_host_shared;
using atlas::UniformGenerate;
using atlas::Float3;

void
expect_vec_near(const Float3& actual, const Float3& expected) {
    EXPECT_NEAR(actual.x, expected.x, atlas::tol);
    EXPECT_NEAR(actual.y, expected.y, atlas::tol);
    EXPECT_NEAR(actual.z, expected.z, atlas::tol);
}

class DummyGenerator final : public Generator {
public:
    explicit DummyGenerator(const Float3& sample = Float3(1.0f, 2.0f, 3.0f))
        : _sample(sample)
        , _operator(UniformGenerate(17u)) {
    }

    Float3
    generate() const override {
        return _sample;
    }

    const Generate&
    generate_operator() const noexcept override {
        return _operator;
    }

    Generate
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
    Float3 _sample;
    Generate _operator;
};

}

TEST(Generator, DerivedImplementationSatisfiesInterface) {
    const DummyGenerator generator;

    expect_vec_near(generator.generate(), Float3(1.0f, 2.0f, 3.0f));
    EXPECT_EQ(generator.type(), GenerateType::uniform);
    EXPECT_NEAR(generator.param0(), 4.0f, atlas::tol);
    EXPECT_NEAR(generator.param1(), 9.0f, atlas::tol);
    EXPECT_EQ(generator.generate_operator().type, GenerateType::uniform);
    EXPECT_EQ(generator.make_generate_operator().type, GenerateType::uniform);
}

TEST(Generator, HostSharedAliasCanOwnDerivedImplementation) {
    GeneratorHostPtr generator = make_host_shared<DummyGenerator>(Float3(3.0f, 2.0f, 1.0f));

    ASSERT_NE(generator, nullptr);
    expect_vec_near(generator->generate(), Float3(3.0f, 2.0f, 1.0f));
}
