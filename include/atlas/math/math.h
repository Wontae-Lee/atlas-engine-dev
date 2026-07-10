#pragma once
/**
 * @file math.h
 * @brief Umbrella header for the Atlas math primitives.
 *
 * Pulls in the small, header-only, host+device value types the engine builds
 * geometry and physics on: scalar constants and helpers, the 3x3 matrix, the
 * quaternion, and the Bool3 / Float3 / Int3 vectors. Include this single header
 * to get the whole set; there is nothing here but the aggregation of the
 * individual headers, which are independently includable.
 */
#include <atlas/math/constants.h>
#include <atlas/math/matrix/float3x3.h>
#include <atlas/math/quaternion.h>
#include <atlas/math/vector/bool3.h>
#include <atlas/math/vector/float3.h>
#include <atlas/math/vector/int3.h>