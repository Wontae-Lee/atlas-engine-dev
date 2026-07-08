#pragma once

#include <atlas/core/macros.h>

#include <new>
#include <type_traits>
#include <utility>

namespace atlas {

template <typename>
inline constexpr bool host_variant_dependent_false = false;

template <typename M>
struct host_member_pointer_traits;

template <typename Class, typename Member>
struct host_member_pointer_traits<Member Class::*> {
    using owner_type   = Class;
    using payload_type = Member;
};

template <auto TagValue, auto Member>
struct HostVariantCase final {
    using owner_type   = typename host_member_pointer_traits<decltype(Member)>::owner_type;
    using payload_type = typename host_member_pointer_traits<decltype(Member)>::payload_type;

    static constexpr auto tag    = TagValue;
    static constexpr auto member = Member;
};

// Host-only counterpart to DeviceVariant: manages a tagged union whose leaves
// may own host-only resources (e.g. DeviceBuffer). Its members are ATLAS_HOST,
// so — unlike DeviceVariant — nvcc never instantiates the union machinery for
// the device and thus never rejects host-only leaf constructors/destructors.
// It is move-based (the leaves are typically move-only).
template <typename Owner,
          typename Tag,
          Tag DefaultTag,
          typename... Cases>
struct HostVariant final {
    ATLAS_HOST static constexpr bool
    contains(const Tag tag) noexcept {
        return ((tag == Cases::tag) || ...);
    }

    ATLAS_HOST static constexpr Tag
    normalize(const Tag tag) noexcept {
        return contains(tag) ? tag : DefaultTag;
    }

    template <typename... Args>
    ATLAS_HOST static void
    construct(Owner& owner, const Tag tag, Args&&... args) {
        owner.type = normalize(tag);
        construct_by_tag<Cases...>(owner, owner.type, std::forward<Args>(args)...);
    }

    template <typename Payload>
    ATLAS_HOST static void
    construct_payload(Owner& owner, Payload&& payload) {
        construct_by_payload<Cases...>(owner, std::forward<Payload>(payload));
    }

    ATLAS_HOST static void
    move_construct(Owner& owner, Owner&& other) {
        owner.type = other.type;
        move_by_tag<Cases...>(owner, other, owner.type);
    }

    ATLAS_HOST static void
    move_assign(Owner& owner, Owner&& other) {
        if (&owner == &other) {
            return;
        }

        destroy(owner);
        move_construct(owner, std::move(other));
    }

    ATLAS_HOST static void
    destroy(Owner& owner) {
        destroy_by_tag<Cases...>(owner, owner.type);
    }

    template <typename Visitor, typename Fallback>
    ATLAS_HOST static Fallback
    visit(const Owner& owner, Visitor&& visitor, Fallback fallback) {
        return visit_by_tag<Cases...>(owner, owner.type, static_cast<Visitor&&>(visitor), fallback);
    }

    template <typename Visitor>
    ATLAS_HOST static void
    apply(Owner& owner, Visitor&& visitor) {
        apply_by_tag<Cases...>(owner, owner.type, static_cast<Visitor&&>(visitor));
    }

    template <typename Visitor>
    ATLAS_HOST static void
    apply(const Owner& owner, Visitor&& visitor) {
        apply_const_by_tag<Cases...>(owner, owner.type, static_cast<Visitor&&>(visitor));
    }

private:
    template <typename Case, typename... Rest, typename... Args>
    ATLAS_HOST static void
    construct_by_tag(Owner& owner, const Tag tag, Args&&... args) {
        if (tag == Case::tag) {
            using Payload = typename Case::payload_type;
            new (&(owner.*Case::member)) Payload(std::forward<Args>(args)...);
            return;
        }

        if constexpr (sizeof...(Rest) > 0) {
            construct_by_tag<Rest...>(owner, tag, std::forward<Args>(args)...);
        } else if constexpr (Case::tag != DefaultTag) {
            construct_by_tag<Cases...>(owner, DefaultTag, std::forward<Args>(args)...);
        }
    }

    template <typename Case, typename... Rest, typename Payload>
    ATLAS_HOST static void
    construct_by_payload(Owner& owner, Payload&& payload) {
        if constexpr (std::is_same_v<std::decay_t<Payload>, typename Case::payload_type>) {
            owner.type          = Case::tag;
            using ActivePayload = typename Case::payload_type;
            new (&(owner.*Case::member)) ActivePayload(std::forward<Payload>(payload));
            return;
        } else if constexpr (sizeof...(Rest) > 0) {
            construct_by_payload<Rest...>(owner, std::forward<Payload>(payload));
        } else {
            static_assert(host_variant_dependent_false<Payload>, "Unsupported HostVariant payload type.");
        }
    }

    template <typename Case, typename... Rest>
    ATLAS_HOST static void
    move_by_tag(Owner& owner, Owner& other, const Tag tag) {
        if (tag == Case::tag) {
            using Payload = typename Case::payload_type;
            new (&(owner.*Case::member)) Payload(std::move(other.*(Case::member)));
            return;
        }

        if constexpr (sizeof...(Rest) > 0) {
            move_by_tag<Rest...>(owner, other, tag);
        }
    }

    template <typename Case, typename... Rest>
    ATLAS_HOST static void
    destroy_by_tag(Owner& owner, const Tag tag) {
        if (tag == Case::tag) {
            using Payload = typename Case::payload_type;
            (owner.*Case::member).~Payload();
            return;
        }

        if constexpr (sizeof...(Rest) > 0) {
            destroy_by_tag<Rest...>(owner, tag);
        }
    }

    template <typename Case, typename... Rest, typename Visitor, typename Fallback>
    ATLAS_HOST static Fallback
    visit_by_tag(const Owner& owner, const Tag tag, Visitor&& visitor, Fallback fallback) {
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
    ATLAS_HOST static void
    apply_by_tag(Owner& owner, const Tag tag, Visitor&& visitor) {
        if (tag == Case::tag) {
            visitor(owner.*(Case::member));
            return;
        }

        if constexpr (sizeof...(Rest) > 0) {
            apply_by_tag<Rest...>(owner, tag, static_cast<Visitor&&>(visitor));
        }
    }

    template <typename Case, typename... Rest, typename Visitor>
    ATLAS_HOST static void
    apply_const_by_tag(const Owner& owner, const Tag tag, Visitor&& visitor) {
        if (tag == Case::tag) {
            visitor(owner.*(Case::member));
            return;
        }

        if constexpr (sizeof...(Rest) > 0) {
            apply_const_by_tag<Rest...>(owner, tag, static_cast<Visitor&&>(visitor));
        }
    }
};

}
