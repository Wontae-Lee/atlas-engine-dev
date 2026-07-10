#pragma once

namespace atlas {

/**
 * @brief Discriminant tag selecting which leaf a `Codec` tagged union holds.
 *
 * The underlying type is fixed to `int` so the value can be trivially copied to
 * the device and serialized. Each enumerator names exactly one leaf type and its
 * corresponding `HostVariantCase` in `codec.h`; the numeric order here is the
 * wire order and must not be reordered once data has been persisted.
 *
 * @see Codec, KnudsenCodec
 */
enum class CodecType : int {

    /// Selects `KnudsenCodec`: buckets a cell's Knudsen number into a solver index.
    knudsen
};

}