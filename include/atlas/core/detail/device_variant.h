#pragma once

#include <atlas/core/macros.h>

#include <new>
#include <type_traits>

namespace atlas::detail {

template <typename>
inline constexpr bool dependent_false_v = false;

template <typename Payload>
struct type_tag final {
    using type = Payload;
};

template <typename M>
struct member_pointer_traits;

template <typename Class, typename Member>
struct member_pointer_traits<Member Class::*> {
    using owner_type   = Class;
    using payload_type = Member;
};

template <auto TagValue, auto Member>
struct DeviceVariantCase final {
    using owner_type   = typename member_pointer_traits<decltype(Member)>::owner_type;
    using payload_type = typename member_pointer_traits<decltype(Member)>::payload_type;

    static constexpr auto tag    = TagValue;
    static constexpr auto member = Member;
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

    template <typename Visitor, typename Fallback>
    ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE static Fallback
    visit(const Owner& owner, Visitor&& visitor, Fallback fallback) noexcept {
        return visit_by_tag<Cases...>(owner, owner.type, static_cast<Visitor&&>(visitor), fallback);
    }

    template <typename Visitor>
    ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE static void
    apply(Owner& owner, Visitor&& visitor) noexcept {
        apply_by_tag<Cases...>(owner, owner.type, static_cast<Visitor&&>(visitor));
    }

    template <typename Visitor>
    ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE static void
    apply(const Owner& owner, Visitor&& visitor) noexcept {
        apply_const_by_tag<Cases...>(owner, owner.type, static_cast<Visitor&&>(visitor));
    }

    template <typename Visitor, typename Fallback>
    ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE static Fallback
    visit_type(const Tag tag, Visitor&& visitor, Fallback fallback) noexcept {
        return visit_type_by_tag<Cases...>(tag, static_cast<Visitor&&>(visitor), fallback);
    }

private:
    template <typename Case, typename... Rest, typename Visitor, typename Fallback>
    ATLAS_ALL_DEVICE static Fallback
    visit_type_by_tag(const Tag tag, Visitor&& visitor, Fallback fallback) noexcept {
        if (tag == Case::tag) {
            return visitor(type_tag<typename Case::payload_type> {});
        }

        if constexpr (sizeof...(Rest) > 0) {
            return visit_type_by_tag<Rest...>(tag, static_cast<Visitor&&>(visitor), fallback);
        } else {
            return fallback;
        }
    }

    template <typename Case, typename... Rest, typename Visitor, typename Fallback>
    ATLAS_ALL_DEVICE static Fallback
    visit_by_tag(const Owner& owner, const Tag tag, Visitor&& visitor, Fallback fallback) noexcept {
        if (tag == Case::tag) {
            return visitor(owner.*(Case::member));
        }

        if constexpr (sizeof...(Rest) > 0) {
            return visit_by_tag<Rest...>(owner, tag, static_cast<Visitor&&>(visitor), fallback);
        } else {
            return fallback;
        }
    }

    template <typename Case, typename... Rest, typename Visitor>
    ATLAS_ALL_DEVICE static void
    apply_by_tag(Owner& owner, const Tag tag, Visitor&& visitor) noexcept {
        if (tag == Case::tag) {
            visitor(owner.*(Case::member));
            return;
        }

        if constexpr (sizeof...(Rest) > 0) {
            apply_by_tag<Rest...>(owner, tag, static_cast<Visitor&&>(visitor));
        }
    }

    template <typename Case, typename... Rest, typename Visitor>
    ATLAS_ALL_DEVICE static void
    apply_const_by_tag(const Owner& owner, const Tag tag, Visitor&& visitor) noexcept {
        if (tag == Case::tag) {
            visitor(owner.*(Case::member));
            return;
        }

        if constexpr (sizeof...(Rest) > 0) {
            apply_const_by_tag<Rest...>(owner, tag, static_cast<Visitor&&>(visitor));
        }
    }

    template <typename Case, typename... Rest, typename... Args>
    ATLAS_ALL_DEVICE static void
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
    ATLAS_ALL_DEVICE static void
    construct_by_payload(Owner& owner, const Payload& payload) noexcept {
        if constexpr (std::is_same_v<Payload, typename Case::payload_type>) {
            owner.type          = Case::tag;
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
    ATLAS_ALL_DEVICE static void
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
    ATLAS_ALL_DEVICE static void
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

template <typename Tag, Tag TagValue, typename Payload>
struct DeviceTypeCase final {
    using payload_type = Payload;

    static constexpr Tag tag = TagValue;
};

template <typename Tag, Tag DefaultTag, typename... Cases>
struct DeviceTypeSwitch final {
    template <typename Visitor, typename Fallback>
    ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE static Fallback
    visit(const Tag tag, Visitor&& visitor, Fallback fallback) noexcept {
        return visit_by_tag<Cases...>(tag, static_cast<Visitor&&>(visitor), fallback);
    }

private:
    template <typename Case, typename... Rest, typename Visitor, typename Fallback>
    ATLAS_ALL_DEVICE static Fallback
    visit_by_tag(const Tag tag, Visitor&& visitor, Fallback fallback) noexcept {
        if (tag == Case::tag) {
            return visitor(type_tag<typename Case::payload_type> {});
        }

        if constexpr (sizeof...(Rest) > 0) {
            return visit_by_tag<Rest...>(tag, static_cast<Visitor&&>(visitor), fallback);
        } else {
            return fallback;
        }
    }
};

}