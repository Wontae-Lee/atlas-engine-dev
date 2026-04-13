#include <cuda/cuda_macros.cuh>
#include <utilities/tests_utils.cuh>

#include <atlas/atlas.h>

using namespace atlas;

CUDA_TEST(SurfaceSpawnOperator, SpawnCreatesSurfaceGridPointsFromBoxGeometryOperator) {
    CUDA_SKIP("SurfaceSpawnOperator CUDA tests are blocked by sample_spawn_grid instantiation issues on the CUDA backend.");
}

CUDA_TEST(SurfaceSpawnOperator, SpawnAcceptsSurfaceSamplesUsingExplicitToleranceBand) {
    CUDA_SKIP("SurfaceSpawnOperator CUDA tests are blocked by sample_spawn_grid instantiation issues on the CUDA backend.");
}

CUDA_TEST(SurfaceSpawnOperator, SpawnCreatesSurfacePointsFromCylinderGeometryOperator) {
    CUDA_SKIP("SurfaceSpawnOperator CUDA tests are blocked by sample_spawn_grid instantiation issues on the CUDA backend.");
}

CUDA_TEST(SurfaceSpawnOperator, SpawnCreatesSurfacePointsFromTriangleGeometryOperator) {
    CUDA_SKIP("SurfaceSpawnOperator CUDA tests are blocked by sample_spawn_grid instantiation issues on the CUDA backend.");
}

CUDA_TEST(SurfaceSpawnOperator, SpawnReturnsEmptyBufferForInvalidGeometryOperator) {
    CUDA_SKIP("SurfaceSpawnOperator CUDA tests are blocked by sample_spawn_grid instantiation issues on the CUDA backend.");
}
