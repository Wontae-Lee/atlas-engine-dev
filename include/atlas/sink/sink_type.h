#pragma once

namespace atlas {

/**
 * @brief Discriminator tag selecting which leaf a @ref Sink tagged-union holds.
 *
 * The underlying type is fixed to `int` so the value is a stable, trivially
 * copyable tag that survives being copied to the device inside a @ref Sink and
 * drives @ref SinkVariant dispatch. Each enumerator names exactly one leaf type
 * and must stay paired with its `DeviceVariantCase` in `sink.h`; the ordering is
 * not otherwise significant. @ref surface is the default-constructed state.
 */
enum class SinkType : int {

    /// Despawn when the particle lies on the unit's boundary surface (@ref SurfaceSink).
    surface,

    /// Despawn when the particle lies inside the unit's solid volume (@ref VolumeSink).
    volume,

    /// Despawn when the particle would cross the unit's surface within one step (@ref TracingSink).
    tracing
};

}