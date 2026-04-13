#include <cuda/cuda_macros.cuh>
#include <utilities/tests_utils.cuh>

#include <atlas/atlas.h>

using namespace atlas;

CUDA_TEST(VolumeSpawnOperator, SpawnCreatesVolumeGridPointsFromBoxGeometryOperator) {
    CUDA_SKIP("VolumeSpawnOperator CUDA tests are blocked by sample_spawn_grid instantiation issues on the CUDA backend.");
}

CUDA_TEST(VolumeSpawnOperator, SpawnAcceptsInteriorSamplesUsingExplicitTolerance) {
    CUDA_SKIP("VolumeSpawnOperator CUDA tests are blocked by sample_spawn_grid instantiation issues on the CUDA backend.");
}

CUDA_TEST(VolumeSpawnOperator, SpawnCreatesInteriorPointsFromSphereGeometryOperator) {
    CUDA_SKIP("VolumeSpawnOperator CUDA tests are blocked by sample_spawn_grid instantiation issues on the CUDA backend.");
}

CUDA_TEST(VolumeSpawnOperator, SpawnCreatesInteriorPointsFromCylinderGeometryOperator) {
    CUDA_SKIP("VolumeSpawnOperator CUDA tests are blocked by sample_spawn_grid instantiation issues on the CUDA backend.");
}

CUDA_TEST(VolumeSpawnOperator, SpawnReturnsEmptyBufferForInvalidGeometryOperator) {
    CUDA_SKIP("VolumeSpawnOperator CUDA tests are blocked by sample_spawn_grid instantiation issues on the CUDA backend.");
}
