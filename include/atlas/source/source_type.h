#pragma once

namespace atlas {

/**
 * @brief Discriminant tag selecting which leaf a `Source` tagged union holds.
 *
 * The underlying type is fixed to `int` so the value can be trivially copied to
 * the device and serialized. Each enumerator names exactly one leaf type and its
 * corresponding `HostVariantCase` in `source.h`; the order here is the wire order
 * and must not be reordered once data has been persisted.
 *
 * @see Source, SurfaceSource, VolumeSource
 */
enum class SourceType : int {

    /// Emits from the geometry's surface shell — see `SurfaceSource`.
    surface,

    /// Emits from the geometry's interior volume — see `VolumeSource`.
    volume
};

}