#pragma once

/**
 * @file host_variant.h
 * @brief Host-only, move-based tagged-union dispatch for buffer-owning leaf types.
 *
 * @c HostVariant is the counterpart to @c DeviceVariant for umbrella types (Source, and
 * any leaf that owns a @c DeviceBuffer) whose leaves are move-only. @c thrust::device_vector
 * has a host-only copy constructor and cannot be copied on the device, so these leaves
 * cannot use the host+device, copy-based @c DeviceVariant. Instead every operation here is
 * @c ATLAS_HOST only and the lifecycle is move-based: @c move_construct / @c move_assign
 * relocate the active member rather than copying it.
 *
 * The storage contract mirrors @c DeviceVariant: the umbrella class owns a @c Tag member
 * @c type and an anonymous @c union of the leaf payloads, and @c HostVariant is a stateless
 * helper that manages the active member by switching on @c type. Because these objects live
 * only on the host, none of the dispatch is ever captured into a device lambda.
 *
 * @note There is deliberately no copy path (no @c copy_construct / @c assign): the leaves
 *       are move-only by construction, which is the entire reason this variant exists.
 * @see device_variant.h for the trivially-copyable, host+device counterpart.
 */

#include <atlas/core/macros.h>

#include <new>
#include <type_traits>
#include <utility>

namespace atlas {

/**
 * @brief Always-false value template used to defer a @c static_assert to instantiation.
 *
 * Named distinctly from @c DeviceVariant's @c dependent_false_v so both headers can be
 * included together without an ODR clash. See @c dependent_false_v for the rationale.
 *
 * @tparam T The type the dependence is anchored to; its identity is irrelevant.
 */
template <typename>
inline constexpr bool host_variant_dependent_false = false;

/**
 * @brief Primary template that decomposes a pointer-to-member type; only the specialization is defined.
 *
 * @tparam M A pointer-to-data-member type of the form @c Member @c Class::*.
 */
template <typename M>
struct host_member_pointer_traits;

/**
 * @brief Specialization extracting the owner and payload types from @c Member @c Class::*.
 *
 * @tparam Class  The class that owns the member (the umbrella type).
 * @tparam Member The type of the pointed-to data member (the leaf payload type).
 */
template <typename Class, typename Member>
struct host_member_pointer_traits<Member Class::*> {
    using owner_type   = Class;  ///< The umbrella class owning the union member.
    using payload_type = Member; ///< The leaf type stored in that union member.
};

/**
 * @brief One arm of a @c HostVariant: binds an enumerator tag to a union member pointer.
 *
 * @tparam TagValue The enumerator selecting this arm (matched against @c Owner::type).
 * @tparam Member   Pointer-to-member into the umbrella's union naming this arm's payload.
 */
template <auto TagValue, auto Member>
struct HostVariantCase final {
    using owner_type   = typename host_member_pointer_traits<decltype(Member)>::owner_type;   ///< Umbrella class type.
    using payload_type = typename host_member_pointer_traits<decltype(Member)>::payload_type; ///< Leaf payload type.

    static constexpr auto tag    = TagValue; ///< Enumerator value selecting this arm.
    static constexpr auto member = Member;   ///< Pointer-to-member locating this arm's storage.
};

/**
 * @brief Stateless, host-only static dispatcher over a move-only tagged union.
 *
 * Manages the active member of an umbrella object's anonymous union by switching on its
 * @c type field, using move semantics throughout so buffer-owning (move-only) leaves are
 * supported. All operations are @c ATLAS_HOST; none may run on the device.
 *
 * @tparam Owner      The umbrella class; must expose a @c Tag member @c type and a union
 *                    holding every @c Cases payload.
 * @tparam Tag        The enum type used to select the active union member.
 * @tparam DefaultTag The tag chosen whenever an unrecognized tag is encountered, keeping
 *                    dispatch total; the corresponding case must always be present.
 * @tparam Cases      The @c HostVariantCase arms, one per union member.
 */
template <typename Owner,
          typename Tag,
          Tag DefaultTag,
          typename... Cases>
struct HostVariant final {
    /**
     * @brief Reports whether @p tag names one of this variant's arms.
     *
     * @param tag The candidate enumerator.
     * @return @c true if some @c Case has @c tag == @p tag, @c false otherwise.
     */
    ATLAS_HOST static constexpr bool
    contains(const Tag tag) noexcept {
        return ((tag == Cases::tag) || ...);
    }

    /**
     * @brief Maps an arbitrary tag to a valid one, collapsing unknown tags to @c DefaultTag.
     *
     * @param tag The candidate enumerator.
     * @return @p tag if @c contains(tag), otherwise @c DefaultTag.
     */
    ATLAS_HOST static constexpr Tag
    normalize(const Tag tag) noexcept {
        return contains(tag) ? tag : DefaultTag;
    }

    /**
     * @brief Activates the union member selected by @p tag, constructing it in place.
     *
     * Sets @c owner.type to the normalized tag, then placement-news the matching payload
     * from @p args (perfectly forwarded). Assumes no member is currently active.
     *
     * @tparam Args Constructor argument types, forwarded to the payload constructor.
     * @param owner The umbrella object whose union is being initialized.
     * @param tag   Requested active tag; normalized to @c DefaultTag if unknown.
     * @param args  Arguments forwarded to the selected payload's constructor.
     */
    template <typename... Args>
    ATLAS_HOST static void
    construct(Owner& owner, const Tag tag, Args&&... args) {
        owner.type = normalize(tag);
        construct_by_tag<Cases...>(owner, owner.type, std::forward<Args>(args)...);
    }

    /**
     * @brief Activates the union member whose payload type matches @p payload and moves it in.
     *
     * The tag is deduced from the (decayed) payload type at compile time. An unsupported
     * payload type is a compile error (see @c construct_by_payload).
     *
     * @tparam Payload The concrete leaf type (possibly a reference); decayed for matching.
     * @param owner   The umbrella object to initialize.
     * @param payload The leaf value forwarded into the union.
     */
    template <typename Payload>
    ATLAS_HOST static void
    construct_payload(Owner& owner, Payload&& payload) {
        construct_by_payload<Cases...>(owner, std::forward<Payload>(payload));
    }

    /**
     * @brief Move-initializes @p owner's active member from @p other's.
     *
     * Assumes @p owner has no active member yet. @c owner.type is copied verbatim from
     * @p other (not normalized) and the matching payload is move-constructed from @p other's.
     * @p other's moved-from payload is left in whatever state its move constructor produces
     * and should only be destroyed afterward.
     *
     * @param owner The freshly-storage target object.
     * @param other The source object to move from.
     * @note If @p other.type names no arm, no member is constructed and @p owner is left
     *       with a tag pointing at an uninitialized union member; callers only ever move
     *       from validly-tagged objects, so this does not arise in practice.
     */
    ATLAS_HOST static void
    move_construct(Owner& owner, Owner&& other) {
        owner.type = other.type;
        move_by_tag<Cases...>(owner, other, owner.type);
    }

    /**
     * @brief Move-assigns @p other into an already-initialized @p owner.
     *
     * Self-move is a no-op. Otherwise the current member is destroyed and then
     * move-constructed from @p other, so the active tag may change.
     *
     * @param owner The target object, assumed to hold a valid active member.
     * @param other The source object to move from.
     */
    ATLAS_HOST static void
    move_assign(Owner& owner, Owner&& other) {
        if (&owner == &other) {
            return;
        }

        destroy(owner);
        move_construct(owner, std::move(other));
    }

    /**
     * @brief Runs the destructor of the currently active union member.
     *
     * After this call the union has no active member; @c owner.type is left unchanged.
     *
     * @param owner The umbrella object whose active payload is destroyed.
     */
    ATLAS_HOST static void
    destroy(Owner& owner) {
        destroy_by_tag<Cases...>(owner, owner.type);
    }

    /**
     * @brief Calls @p visitor on the active payload and returns its result.
     *
     * @tparam Visitor  Callable accepting a @c const reference to the active leaf.
     * @tparam Fallback The return type; also the value returned when no arm matches.
     * @param owner    The object to inspect.
     * @param visitor  The callable invoked with the active leaf.
     * @param fallback Value returned if @c owner.type matches no arm.
     * @return @c visitor(active leaf), or @p fallback when nothing matched.
     */
    template <typename Visitor, typename Fallback>
    ATLAS_HOST static Fallback
    visit(const Owner& owner, Visitor&& visitor, Fallback fallback) {
        return visit_by_tag<Cases...>(owner, owner.type, static_cast<Visitor&&>(visitor), fallback);
    }

    /**
     * @brief Invokes @p visitor on the active payload for a mutating side effect (no result).
     *
     * If @c owner.type matches no arm the call is silently skipped.
     *
     * @tparam Visitor Callable accepting a mutable reference to the active leaf.
     * @param owner   The object whose active leaf is passed to @p visitor.
     * @param visitor The callable applied to the active leaf.
     */
    template <typename Visitor>
    ATLAS_HOST static void
    apply(Owner& owner, Visitor&& visitor) {
        apply_by_tag<Cases...>(owner, owner.type, static_cast<Visitor&&>(visitor));
    }

    /**
     * @brief Const overload of @c apply: invokes @p visitor on the active payload read-only.
     *
     * @tparam Visitor Callable accepting a @c const reference to the active leaf.
     * @param owner   The object whose active leaf is passed to @p visitor.
     * @param visitor The callable applied to the active leaf.
     */
    template <typename Visitor>
    ATLAS_HOST static void
    apply(const Owner& owner, Visitor&& visitor) {
        apply_const_by_tag<Cases...>(owner, owner.type, static_cast<Visitor&&>(visitor));
    }

private:
    /**
     * @brief Recursive implementation of @c construct: placement-news the matching payload.
     *
     * When @p tag matches no arm, recursion restarts from the full @c Cases list at
     * @c DefaultTag; the @c Case::tag != DefaultTag guard on the terminal step prevents
     * infinite recursion once the default arm itself has failed to match.
     *
     * @tparam Case The arm being tested this step.
     * @tparam Rest The remaining arms.
     * @tparam Args Constructor argument types, forwarded to the payload.
     * @param owner The object whose union member is constructed.
     * @param tag   The tag selecting the member.
     * @param args  Arguments forwarded to the payload constructor.
     */
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

    /**
     * @brief Recursive implementation of @c construct_payload: matches on decayed payload type.
     *
     * Sets @c owner.type to the arm whose @c payload_type equals @c std::decay_t<Payload>
     * and move/forward-constructs it. If no arm matches, a dependent @c static_assert
     * rejects the type at compile time.
     *
     * @tparam Case    The arm being tested this step.
     * @tparam Rest    The remaining arms.
     * @tparam Payload The forwarding-reference payload type being stored.
     * @param owner   The object to initialize.
     * @param payload The value forwarded into the union.
     */
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

    /**
     * @brief Recursive implementation of @c move_construct: move-news the matching payload.
     *
     * Unlike @c DeviceVariant::copy_by_tag there is no @c DefaultTag fallback here: if
     * @p tag matches no arm, nothing is constructed. Callers only move from validly-tagged
     * objects, so the tag always matches an arm in practice.
     *
     * @tparam Case The arm being tested this step.
     * @tparam Rest The remaining arms.
     * @param owner The target object.
     * @param other The source object, moved from member-wise.
     * @param tag   The active tag driving the search.
     */
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

    /**
     * @brief Recursive implementation of @c destroy: destructs the matching payload.
     *
     * No-ops if @p tag matches no arm (nothing to destroy).
     *
     * @tparam Case The arm being tested this step.
     * @tparam Rest The remaining arms.
     * @param owner The object whose active member is destroyed.
     * @param tag   The active tag identifying the member.
     */
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

    /**
     * @brief Recursive implementation of @c visit: dereferences the matching union member.
     *
     * @tparam Case     The arm being tested this step.
     * @tparam Rest     The remaining arms.
     * @tparam Visitor  Callable accepting a @c const reference to the leaf.
     * @tparam Fallback Return type and no-match value.
     * @param owner    The object being inspected.
     * @param tag      The active tag driving the search.
     * @param visitor  Callable invoked on the matched leaf.
     * @param fallback Returned when no arm matches.
     * @return The visitor result, or @p fallback.
     */
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

    /**
     * @brief Recursive implementation of the mutable @c apply.
     *
     * @tparam Case    The arm being tested this step.
     * @tparam Rest    The remaining arms.
     * @tparam Visitor Callable accepting a mutable reference to the leaf.
     * @param owner   The object whose matched leaf is visited.
     * @param tag     The active tag driving the search.
     * @param visitor Callable invoked on the matched leaf.
     */
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

    /**
     * @brief Recursive implementation of the const @c apply.
     *
     * @tparam Case    The arm being tested this step.
     * @tparam Rest    The remaining arms.
     * @tparam Visitor Callable accepting a @c const reference to the leaf.
     * @param owner   The object whose matched leaf is visited.
     * @param tag     The active tag driving the search.
     * @param visitor Callable invoked on the matched leaf.
     */
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

} // namespace atlas