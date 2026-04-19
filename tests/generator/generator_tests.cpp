#include "../utilities/tests_utils.h"

#include <atlas/generator/generate_operator.h>
#include <atlas/generator/generator.h>

#include <testkit/testkit.h>

namespace {

using T = float;
using Vec3 = atlas::Vector3<T>;

class DummyGenerator final : public atlas::fluid::Generator<T> {
public:
    explicit DummyGenerator(Vec3 sample = Vec3(1, 2, 3))
        : _sample(sample)
        , _operator(atlas::fluid::UniformGenerateOperator<T>(17u)) {}

    Vec3
    generate() const override {
        return _sample;
    }

    const atlas::fluid::GenerateOperator<T>&
    generate_operator() const noexcept override {
        return _operator;
    }

    atlas::fluid::GenerateOperator<T>
    make_generate_operator() const noexcept override {
        return _operator;
    }

    T
    param0() const noexcept override {
        return 4.0f;
    }

    T
    param1() const noexcept override {
        return 9.0f;
    }

    atlas::fluid::GenerateType
    type() const noexcept override {
        return atlas::fluid::GenerateType::uniform;
    }

private:
    Vec3 _sample;
    atlas::fluid::GenerateOperator<T> _operator;
};

constexpr T kEps = static_cast<T>(1e-5);

} // namespace

TEST(Generator, DerivedImplementationSatisfiesInterface) {
    const DummyGenerator generator;

    EXPECT_TRUE(atlas::test::vec_near(generator.generate(), Vec3(1, 2, 3), kEps));
    EXPECT_EQ(generator.type(), atlas::fluid::GenerateType::uniform);
    EXPECT_NEAR(generator.param0(), 4.0f, kEps);
    EXPECT_NEAR(generator.param1(), 9.0f, kEps);
    EXPECT_EQ(generator.generate_operator().type, atlas::fluid::GenerateType::uniform);
    EXPECT_EQ(generator.make_generate_operator().type, atlas::fluid::GenerateType::uniform);
}

TEST(Generator, HostSharedAliasCanOwnDerivedImplementation) {
    atlas::GeneratorHostPtr<T> generator = atlas::make_host_shared<DummyGenerator>(Vec3(3, 2, 1));

    ASSERT_NE(generator, nullptr);
    EXPECT_TRUE(atlas::test::vec_near(generator->generate(), Vec3(3, 2, 1), kEps));
}
