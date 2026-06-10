#include <atlas/collider/interaction/maxwellian_surface_interaction.h>
#include <atlas/collider/interaction/surface_interaction_kernel.h>
#include <atlas/math/math.h>
#include <atlas/sampling/sampling.h>

#include <benchmark/benchmark.h>

#include <array>
#include <cmath>
#include <cstdint>
#include <limits>

namespace {

using Scalar = double;
using Vector = atlas::Vector3<Scalar>;

constexpr Scalar kSurfaceTemperature = 293.0;
constexpr Scalar kNitrogenMass       = 4.651734e-26;
constexpr Scalar kMomentumAcc      = 1.0;
constexpr int kSampleCount           = 4096;

struct SampleSet final {
    Vector incident;
    Vector normal;
    Vector tangent_seed;
    Scalar branch;
    Scalar perpendicular;
    Scalar theta;
    Scalar tangent;
};

Scalar
unit_sample(const int index, const std::uint64_t seed) {
    return atlas::sampling::sample_hashed_unit_interval<Scalar>(index, seed);
}

std::array<SampleSet, kSampleCount>
make_samples() {
    std::array<SampleSet, kSampleCount> samples {};

    for (int i = 0; i < kSampleCount; ++i) {
        const Scalar nx = unit_sample(i, 0x21u) * Scalar(2) - Scalar(1);
        const Scalar ny = unit_sample(i, 0x43u) * Scalar(2) - Scalar(1);
        const Scalar nz = unit_sample(i, 0x65u) * Scalar(2) - Scalar(1);
        Vector normal(nx, ny, nz);
        normal = atlas::math::normalize(normal);

        samples[i] = SampleSet {
            Vector(
                Scalar(900) * (unit_sample(i, 0x87u) - Scalar(0.5)),
                Scalar(900) * (unit_sample(i, 0xa9u) - Scalar(0.5)),
                Scalar(900) * (unit_sample(i, 0xcbu) - Scalar(0.5))),
            normal,
            Vector(
                unit_sample(i, 0xedu),
                unit_sample(i, 0x10fu),
                unit_sample(i, 0x131u)),
            unit_sample(i, 0x153u),
            unit_sample(i, 0x175u),
            unit_sample(i, 0x197u),
            unit_sample(i, 0x1b9u),
        };
    }

    return samples;
}

Vector
sparta_reference(const atlas::MaxwellianSurfaceInteraction<Scalar>& interaction,
                 const SampleSet& sample) {
    if (sample.branch > interaction.momentum_acc()) {
        return atlas::math::reflected(sample.incident, sample.normal);
    }

    const Scalar vrm      = std::sqrt(Scalar(2) * atlas::boltzmann_constant
                                      * interaction.temperature() / interaction.molecular_mass());
    const Scalar vperp    = vrm * std::sqrt(-std::log(std::max(sample.perpendicular, atlas::eps)));
    const Scalar theta    = Scalar(2) * atlas::pi * sample.theta;
    const Scalar vtangent = vrm * std::sqrt(-std::log(std::max(sample.tangent, atlas::eps)));
    const Scalar vtan1    = vtangent * std::sin(theta);
    const Scalar vtan2    = vtangent * std::cos(theta);

    const Scalar dot = atlas::math::dot(sample.incident, sample.normal);
    Vector tangent1 = sample.incident - sample.normal * dot;

    if (tangent1.length_squared() == Scalar(0)) {
        tangent1 = atlas::math::cross(sample.normal, sample.tangent_seed);
        if (tangent1.length_squared() <= atlas::tol) {
            const Vector axis = (std::abs(sample.normal.x) < Scalar(0.9))
                ? Vector(Scalar(1), Scalar(0), Scalar(0))
                : Vector(Scalar(0), Scalar(1), Scalar(0));
            tangent1 = atlas::math::cross(sample.normal, axis);
        }
    }

    tangent1 = atlas::math::normalize(tangent1);
    const Vector tangent2 = atlas::math::cross(sample.normal, tangent1);

    return sample.normal * vperp + tangent1 * vtan1 + tangent2 * vtan2;
}

bool
nearly_equal(const Vector& lhs, const Vector& rhs) {
    const Vector delta = lhs - rhs;
    return delta.length() <= Scalar(1e-10) * std::max(Scalar(1), rhs.length());
}

bool
verify_sparta_equivalence(const atlas::MaxwellianSurfaceInteraction<Scalar>& interaction,
                          const std::array<SampleSet, kSampleCount>& samples) {
    const atlas::SurfaceInteractionKernel<Scalar> kernel(interaction);
    if (kernel.type != atlas::SurfaceInteractionType::maxwellian) {
        return false;
    }

    for (const SampleSet& sample : samples) {
        const Vector atlas_result = kernel.maxwellian.sample(
            sample.incident,
            sample.normal,
            sample.branch,
            sample.perpendicular,
            sample.theta,
            sample.tangent,
            sample.tangent_seed);
        const Vector reference_result = sparta_reference(interaction, sample);

        if (!nearly_equal(atlas_result, reference_result)) {
            return false;
        }
    }

    return true;
}

void
BM_MaxwellianSurfaceInteraction(benchmark::State& state) {
    const auto samples = make_samples();
    const auto interaction = atlas::MaxwellianSurfaceInteraction<Scalar>::builder()
                                 .with_temperature(kSurfaceTemperature)
                                 .with_molecular_mass(kNitrogenMass)
                                 .with_momentum_acc(kMomentumAcc)
                                 .build();
    const atlas::SurfaceInteractionKernel<Scalar> kernel(interaction);

    if (!verify_sparta_equivalence(interaction, samples)) {
        state.SkipWithError("Atlas Maxwellian surface interaction diverged from SPARTA reference formula.");
        return;
    }

    std::size_t index = 0;
    for (auto _ : state) {
        const SampleSet& sample = samples[index++ % samples.size()];
        const Vector result = kernel(sample.incident, sample.normal);
        benchmark::DoNotOptimize(result);
    }
}

void
BM_SpartaReferenceDiffuseFormula(benchmark::State& state) {
    const auto samples = make_samples();
    const auto interaction = atlas::MaxwellianSurfaceInteraction<Scalar>::builder()
                                 .with_temperature(kSurfaceTemperature)
                                 .with_molecular_mass(kNitrogenMass)
                                 .with_momentum_acc(kMomentumAcc)
                                 .build();

    std::size_t index = 0;
    for (auto _ : state) {
        const Vector result = sparta_reference(interaction, samples[index++ % samples.size()]);
        benchmark::DoNotOptimize(result);
    }
}

} // namespace

BENCHMARK(BM_MaxwellianSurfaceInteraction);
BENCHMARK(BM_SpartaReferenceDiffuseFormula);
