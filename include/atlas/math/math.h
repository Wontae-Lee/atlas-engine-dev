/**
 * @file  math.h
 * @brief Umbrella header that pulls in the full atlas math module.
 *
 * Includes constants, vector (Float3, Int3, Bool3), matrix (Float3x3),
 * and quaternion types in one shot for consumers that don't need to
 * cherry-pick individual headers.
 */

#pragma once
#include <atlas/math/constants.h>
#include <atlas/math/matrix/float3x3.h>
#include <atlas/math/quaternion.h>
#include <atlas/math/vector/bool3.h>
#include <atlas/math/vector/float3.h>
#include <atlas/math/vector/int3.h>
