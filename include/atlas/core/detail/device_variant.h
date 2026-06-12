#pragma once

/**
 * @file device_variant.h
 * @brief Shared lifecycle helpers for device-portable tagged unions.
 */

#include <atlas/core/macros.h>

#include <new>
#include <type_traits>

namespace atlas::detail {

template <typename>
inline constexpr bool dependent_false_v = false;

template <typename Owner,
          typename Tag,
          Tag TagValue,
          typename Payload,
          Payload Owner::* Member>
struct DeviceVariantCase final {
    using payload_type = Payload;

    static constexpr Tag tag = TagValue;
    static constexpr Payload Owner::* member = Member;
};

template <typename Owner,
          typename Tag,
          Tag DefaultTag,
          typename... Cases>
struct DeviceVariant final {
    ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE static constexpr bool
    contains(const Tag tag) noexcept {
        return ((tag == Cases::tag) || ...);
    }

    ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE static constexpr Tag
    normalize(const Tag tag) noexcept {
        return contains(tag) ? tag : DefaultTag;
    }

    template <typename... Args>
    ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE static void
    construct(Owner& owner, const Tag tag, const Args&... args) noexcept {
        owner.type = normalize(tag);
        construct_by_tag<Cases...>(owner, owner.type, args...);
    }

    template <typename Payload>
    ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE static void
    construct_payload(Owner& owner, const Payload& payload) noexcept {
        construct_by_payload<Payload, Cases...>(owner, payload);
    }

    ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE static void
    copy_construct(Owner& owner, const Owner& other) noexcept {
        owner.type = normalize(other.type);
        copy_by_tag<Cases...>(owner, other, owner.type);
    }

    ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE static void
    assign(Owner& owner, const Owner& other) noexcept {
        if (&owner == &other) return;

        destroy(owner);
        copy_construct(owner, other);
    }

    ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE static void
    destroy(Owner& owner) noexcept {
        destroy_by_tag<Cases...>(owner, owner.type);
    }

private:
    template <typename Case, typename... Rest, typename... Args>
    ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE static void
    construct_by_tag(Owner& owner, const Tag tag, const Args&... args) noexcept {
        if (tag == Case::tag) {
            using Payload = typename Case::payload_type;
            new (&(owner.*Case::member)) Payload(args...);
            return;
        }

        if constexpr (sizeof...(Rest) > 0) {
            construct_by_tag<Rest...>(owner, tag, args...);
        } else if constexpr (Case::tag != DefaultTag) {
            construct_by_tag<Cases...>(owner, DefaultTag, args...);
        }
    }

    template <typename Payload, typename Case, typename... Rest>
    ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE static void
    construct_by_payload(Owner& owner, const Payload& payload) noexcept {
        if constexpr (std::is_same_v<Payload, typename Case::payload_type>) {
            owner.type = Case::tag;
            using ActivePayload = typename Case::payload_type;
            new (&(owner.*Case::member)) ActivePayload(payload);
            return;
        } else if constexpr (sizeof...(Rest) > 0) {
            construct_by_payload<Payload, Rest...>(owner, payload);
        } else {
            static_assert(dependent_false_v<Payload>, "Unsupported DeviceVariant payload type.");
        }
    }

    template <typename Case, typename... Rest>
    ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE static void
    copy_by_tag(Owner& owner, const Owner& other, const Tag tag) noexcept {
        if (tag == Case::tag) {
            using Payload = typename Case::payload_type;
            new (&(owner.*Case::member)) Payload(other.*(Case::member));
            return;
        }

        if constexpr (sizeof...(Rest) > 0) {
            copy_by_tag<Rest...>(owner, other, tag);
        } else if constexpr (Case::tag != DefaultTag) {
            construct_by_tag<Cases...>(owner, DefaultTag);
        }
    }

    template <typename Case, typename... Rest>
    ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE static void
    destroy_by_tag(Owner& owner, const Tag tag) noexcept {
        if (tag == Case::tag) {
            using Payload = typename Case::payload_type;
            (owner.*Case::member).~Payload();
            return;
        }

        if constexpr (sizeof...(Rest) > 0) {
            destroy_by_tag<Rest...>(owner, tag);
        }
    }
};

} // namespace atlas::detail
