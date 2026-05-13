#include <atlas/fluid/fluid.h>
#include <atlas/fluid/fluid_state.h>
#include <atlas/material/material_properties.h>
#include <atlas/memory/raw_pointer_cast.h>
#include <atlas/parallel/parallel_for.h>
#include <atlas/system/system.h>

#include <benchmark/benchmark.h>

#include <cstddef>
#include <cstdint>
#include <memory>

namespace {

using Scalar = float;
using Vector = atlas::Vector3<Scalar>;

atlas::DeviceBuffer<Vector>
make_vectors(const std::size_t size, const Scalar scale) {
    atlas::DeviceBuffer<Vector> values(size);
    for (std::size_t i = 0; i < size; ++i) {
        const auto value = static_cast<Scalar>(i % 1024) * scale;
        values[i] = Vector(value, value + Scalar(1), value + Scalar(2));
    }
    return values;
}

atlas::HostBuffer<atlas::MaterialProperties<Scalar>>
make_properties() {
    atlas::HostBuffer<atlas::MaterialProperties<Scalar>> properties(1);
    properties[0] = atlas::MaterialProperties<Scalar>::builder()
                        .with_type(atlas::MaterialType::Molecule)
                        .with_mass(Scalar(4.651734e-26))
                        .with_molecular_mass(Scalar(4.651734e-26))
                        .with_collision_diameter(Scalar(4.17e-10))
                        .build();
    return properties;
}

atlas::HostBuffer<atlas::GeneratorHostPtr<Scalar>>
make_generators() {
    return atlas::HostBuffer<atlas::GeneratorHostPtr<Scalar>>(1);
}

atlas::FluidHostPtr<Scalar>
make_fluid(const std::size_t particle_count) {
    auto fluid = atlas::Fluid<Scalar>::builder()
                     .with_buffer_size(particle_count)
                     .with_properties(make_properties())
                     .with_generators(make_generators())
                     .make_host_shared();

    fluid->set_state<atlas::fluid::FluidPositionState<Scalar>>(
        std::make_unique<atlas::fluid::FluidPositionState<Scalar>>(
            make_vectors(particle_count, Scalar(0.001))));
    fluid->set_state<atlas::fluid::FluidVelocityState<Scalar>>(
        std::make_unique<atlas::fluid::FluidVelocityState<Scalar>>(
            make_vectors(particle_count, Scalar(0.0001))));
    fluid->set_particle_count(particle_count);

    return fluid;
}

void
BM_DeviceBufferVectorResize(benchmark::State& state) {
    const auto size = static_cast<std::size_t>(state.range(0));

    for (auto _ : state) {
        atlas::DeviceBuffer<Vector> values;
        values.resize(size);
        benchmark::DoNotOptimize(values.data());
        benchmark::ClobberMemory();
    }

    state.SetItemsProcessed(state.iterations() * static_cast<std::int64_t>(size));
}

void
BM_FluidBuildDefaultStates(benchmark::State& state) {
    const auto particle_count = static_cast<std::size_t>(state.range(0));
    const auto properties = make_properties();
    const auto generators = make_generators();

    for (auto _ : state) {
        auto fluid = atlas::Fluid<Scalar>::builder()
                         .with_buffer_size(particle_count)
                         .with_properties(properties)
                         .with_generators(generators)
                         .make_host_shared();
        benchmark::DoNotOptimize(fluid.get());
        benchmark::ClobberMemory();
    }

    state.SetItemsProcessed(state.iterations() * static_cast<std::int64_t>(particle_count));
}

void
BM_ParallelForSerialPositionUpdate(benchmark::State& state) {
    const auto particle_count = static_cast<int>(state.range(0));
    auto positions = make_vectors(static_cast<std::size_t>(particle_count), Scalar(0.001));
    auto velocities = make_vectors(static_cast<std::size_t>(particle_count), Scalar(0.0001));
    auto* positions_ptr = atlas::raw_pointer_cast(positions.data());
    const auto* velocities_ptr = atlas::raw_pointer_cast(velocities.data());
    constexpr Scalar dt = Scalar(0.01);

    for (auto _ : state) {
        atlas::parallel_for<atlas::ExecutionPolicy::serial>(
            0,
            particle_count,
            [positions_ptr, velocities_ptr] ATLAS_DEVICE(const int i) {
                positions_ptr[i] += velocities_ptr[i] * dt;
            });
        benchmark::DoNotOptimize(positions_ptr);
        benchmark::ClobberMemory();
    }

    state.SetItemsProcessed(state.iterations() * static_cast<std::int64_t>(particle_count));
}

void
BM_ParallelForDevicePositionUpdate(benchmark::State& state) {
    const auto particle_count = static_cast<int>(state.range(0));
    auto positions = make_vectors(static_cast<std::size_t>(particle_count), Scalar(0.001));
    auto velocities = make_vectors(static_cast<std::size_t>(particle_count), Scalar(0.0001));
    auto* positions_ptr = atlas::raw_pointer_cast(positions.data());
    const auto* velocities_ptr = atlas::raw_pointer_cast(velocities.data());
    constexpr Scalar dt = Scalar(0.01);

    for (auto _ : state) {
        atlas::parallel_for<atlas::ExecutionPolicy::device>(
            0,
            particle_count,
            [positions_ptr, velocities_ptr] ATLAS_DEVICE(const int i) {
                positions_ptr[i] += velocities_ptr[i] * dt;
            });
        benchmark::DoNotOptimize(positions_ptr);
        benchmark::ClobberMemory();
    }

    state.SetItemsProcessed(state.iterations() * static_cast<std::int64_t>(particle_count));
}

void
BM_SystemUpdateTimeIntegration(benchmark::State& state) {
    const auto particle_count = static_cast<std::size_t>(state.range(0));

    auto fluid = make_fluid(particle_count);
    auto system = atlas::system::System<Scalar>::builder()
                      .with_fluid(fluid)
                      .with_dt(Scalar(0.01))
                      .build();

    for (auto _ : state) {
        system.update();
        benchmark::DoNotOptimize(system.fluid().get());
        benchmark::ClobberMemory();
    }

    state.SetItemsProcessed(state.iterations() * static_cast<std::int64_t>(particle_count));
}

void
BM_FluidStateLookup(benchmark::State& state) {
    auto fluid = make_fluid(static_cast<std::size_t>(state.range(0)));

    for (auto _ : state) {
        auto* position_state = fluid->state<atlas::fluid::FluidPositionState<Scalar>>();
        auto* velocity_state = fluid->state<atlas::fluid::FluidVelocityState<Scalar>>();
        benchmark::DoNotOptimize(position_state);
        benchmark::DoNotOptimize(velocity_state);
    }
}

} // namespace

BENCHMARK(BM_DeviceBufferVectorResize)->RangeMultiplier(4)->Range(1 << 10, 1 << 20);
BENCHMARK(BM_FluidBuildDefaultStates)->RangeMultiplier(4)->Range(1 << 10, 1 << 20);
BENCHMARK(BM_ParallelForSerialPositionUpdate)->RangeMultiplier(4)->Range(1 << 10, 1 << 20);
BENCHMARK(BM_ParallelForDevicePositionUpdate)->RangeMultiplier(4)->Range(1 << 10, 1 << 20);
BENCHMARK(BM_SystemUpdateTimeIntegration)->RangeMultiplier(4)->Range(1 << 10, 1 << 20);
BENCHMARK(BM_FluidStateLookup)->Arg(1 << 20);
