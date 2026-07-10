#pragma once

/**
 * @file device_variant.h
 * @brief Host+device tagged-union dispatch for trivially-copyable leaf types.
 *
 * @c DeviceVariant is the shared machinery behind every umbrella type whose leaves are
 * trivially copyable (Geometry, Material, Collider, Codec, DsmcKernel, ...). The umbrella
 * class itself owns the storage: a @c Tag-typed data member named @c type and an anonymous
 * @c union of the leaf payloads. @c DeviceVariant is a stateless helper parameterized on
 * that class; its static methods manage the union's active member (placement-new,
 * destroy, copy, visit) by dispatching on @c type.
 *
 * Every method is @c ATLAS_ALL_DEVICE, so the same dispatch runs on the host during setup
 * and inside a device kernel after the umbrella object has been captured by value into a
 * device lambda. This host+device capturability is the whole reason the leaves must be
 * trivially copyable and store no @c DeviceBuffer; move-only, buffer-owning leaves use the
 * host-only @c HostVariant instead.
 *
 * @note Each @c Case is a @c DeviceVariantCase pairing an enumerator @c tag with a
 *       pointer-to-member into the union. @c DefaultTag is the fallback used whenever an
 *       out-of-range tag is seen, so dispatch is always defined even for corrupt input.
 * @see host_variant.h for the move-based, host-only counterpart.
 */

#include <atlas/core/macros.h>

#include <new>
#include <type_traits>

namespace atlas {

/**
 * @brief Always-false value template used to defer a @c static_assert to instantiation.
 *
 * Because it depends on @p T, a @c static_assert(dependent_false_v<T>) only fires when the
 * enclosing template is actually instantiated for that @p T, letting an otherwise-unreached
 * @c else branch reject an unsupported payload with a readable message.
 *
 * @tparam T The type the dependence is anchored to; its identity is irrelevant.
 */
template <typename>
inline constexpr bool dependent_false_v = false;

/**
 * @brief Carries a payload type as a value so it can be passed to a visitor.
 *
 * Used by the type-only visitors (@c visit_type / @c DeviceTypeSwitch::visit): the visitor
 * receives a @c type_tag instance and recovers the payload type from @c type_tag::type,
 * without any payload object needing to exist.
 *
 * @tparam Payload The type being carried.
 */
template <typename Payload>
struct type_tag final {
    using type = Payload; ///< The payload type this tag stands for.
};

/**
 * @brief Primary template that decomposes a pointer-to-member type; only the specialization is defined.
 *
 * @tparam M A pointer-to-data-member type of the form @c Member @c Class::*.
 */
template <typename M>
struct member_pointer_traits;

/**
 * @brief Specialization extracting the owner and payload types from @c Member @c Class::*.
 *
 * @tparam Class  The class that owns the member (the umbrella type).
 * @tparam Member The type of the pointed-to data member (the leaf payload type).
 */
template <typename Class, typename Member>
struct member_pointer_traits<Member Class::*> {
    using owner_type   = Class;  ///< The umbrella class owning the union member.
    using payload_type = Member; ///< The leaf type stored in that union member.
};

/**
 * @brief One arm of a @c DeviceVariant: binds an enumerator tag to a union member pointer.
 *
 * @tparam TagValue The enumerator selecting this arm (matched against @c Owner::type).
 * @tparam Member   Pointer-to-member into the umbrella's union naming this arm's payload.
 */
template <auto TagValue, auto Member>
struct DeviceVariantCase final {
    using owner_type   = typename member_pointer_traits<decltype(Member)>::owner_type;   ///< Umbrella class type.
    using payload_type = typename member_pointer_traits<decltype(Member)>::payload_type; ///< Leaf payload type.

    static constexpr auto tag    = TagValue; ///< Enumerator value selecting this arm.
    static constexpr auto member = Member;   ///< Pointer-to-member locating this arm's storage.
};

/**
 * @brief Stateless static dispatcher over a trivially-copyable tagged union.
 *
 * Manages the active member of an umbrella object's anonymous union by switching on its
 * @c type field. All operations are @c ATLAS_ALL_DEVICE and @c noexcept so they are usable
 * verbatim from host setup code and from device kernels.
 *
 * @tparam Owner      The umbrella class; must expose a @c Tag member @c type and a union
 *                    holding every @c Cases payload.
 * @tparam Tag        The enum type used to select the active union member.
 * @tparam DefaultTag The tag chosen whenever an unrecognized tag is encountered, keeping
 *                    dispatch total; the corresponding case must always be present.
 * @tparam Cases      The @c DeviceVariantCase arms, one per union member.
 */
template <typename Owner,
          typename Tag,
          Tag DefaultTag,
          typename... Cases>
struct DeviceVariant final {
    /**
     * @brief Reports whether @p tag names one of this variant's arms.
     *
     * @param tag The candidate enumerator.
     * @return @c true if some @c Case has @c tag == @p tag, @c false otherwise.
     */
    ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE static constexpr bool
    contains(const Tag tag) noexcept {
        return ((tag == Cases::tag) || ...);
    }

    /**
     * @brief Maps an arbitrary tag to a valid one, collapsing unknown tags to @c DefaultTag.
     *
     * This keeps every subsequent dispatch total: a corrupt or default-initialized @c type
     * never selects a non-existent union member.
     *
     * @param tag The candidate enumerator.
     * @return @p tag if @c contains(tag), otherwise @c DefaultTag.
     */
    ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE static constexpr Tag
    normalize(const Tag tag) noexcept {
        return contains(tag) ? tag : DefaultTag;
    }

    /**
     * @brief Activates the union member selected by @p tag, constructing it in place.
     *
     * Sets @c owner.type to the normalized tag, then placement-news the matching payload
     * from @p args. The caller is responsible for ensuring no member is currently active
     * (i.e. this is initialization, not reassignment); use @c assign to replace a live one.
     *
     * @tparam Args Constructor argument types forwarded to the payload (by const reference).
     * @param owner The umbrella object whose union is being initialized.
     * @param tag   Requested active tag; normalized to @c DefaultTag if unknown.
     * @param args  Arguments forwarded to the selected payload's constructor.
     */
    template <typename... Args>
    ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE static void
    construct(Owner& owner, const Tag tag, const Args&... args) noexcept {
        owner.type = normalize(tag);
        construct_by_tag<Cases...>(owner, owner.type, args...);
    }

    /**
     * @brief Activates the union member whose payload type matches @p payload and copies it in.
     *
     * The tag is deduced from the payload type at compile time, so callers construct a
     * variant directly from a concrete leaf without naming its enumerator. An unsupported
     * payload type is a compile error (see @c construct_by_payload).
     *
     * @tparam Payload The concrete leaf type; must equal exactly one arm's @c payload_type.
     * @param owner   The umbrella object to initialize.
     * @param payload The leaf value to copy into the union.
     */
    template <typename Payload>
    ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE static void
    construct_payload(Owner& owner, const Payload& payload) noexcept {
        construct_by_payload<Payload, Cases...>(owner, payload);
    }

    /**
     * @brief Copy-initializes @p owner's active member from @p other's.
     *
     * Assumes @p owner has no active member yet. @c owner.type is set to the normalized tag
     * of @p other, then the matching payload is copy-constructed from @p other's.
     *
     * @param owner The freshly-storage target object.
     * @param other The source object to copy from.
     */
    ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE static void
    copy_construct(Owner& owner, const Owner& other) noexcept {
        owner.type = normalize(other.type);
        copy_by_tag<Cases...>(owner, other, owner.type);
    }

    /**
     * @brief Copy-assigns @p other into an already-initialized @p owner.
     *
     * Self-assignment is a no-op. Otherwise the current member is destroyed and then
     * copy-constructed from @p other, so the active tag may change.
     *
     * @param owner The target object, assumed to hold a valid active member.
     * @param other The source object to copy from.
     */
    ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE static void
    assign(Owner& owner, const Owner& other) noexcept {
        if (&owner == &other) return;

        destroy(owner);
        copy_construct(owner, other);
    }

    /**
     * @brief Runs the destructor of the currently active union member.
     *
     * After this call the union has no active member; @c owner.type is left unchanged and
     * must not be used to access storage until a member is reconstructed.
     *
     * @param owner The umbrella object whose active payload is destroyed.
     */
    ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE static void
    destroy(Owner& owner) noexcept {
        destroy_by_tag<Cases...>(owner, owner.type);
    }

    /**
     * @brief Calls @p visitor on the active payload and returns its result.
     *
     * @tparam Visitor  Callable accepting a @c const reference to the active leaf.
     * @tparam Fallback The return type; also the value returned when no arm matches.
     * @param owner    The object to inspect.
     * @param visitor  The callable invoked with the active leaf.
     * @param fallback Value returned if @c owner.type matches no arm (e.g. corrupt tag).
     * @return @c visitor(active leaf), or @p fallback when nothing matched.
     */
    template <typename Visitor, typename Fallback>
    ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE static Fallback
    visit(const Owner& owner, Visitor&& visitor, Fallback fallback) noexcept {
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
    ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE static void
    apply(Owner& owner, Visitor&& visitor) noexcept {
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
    ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE static void
    apply(const Owner& owner, Visitor&& visitor) noexcept {
        apply_const_by_tag<Cases...>(owner, owner.type, static_cast<Visitor&&>(visitor));
    }

    /**
     * @brief Visits by tag alone, passing a @c type_tag rather than any live payload.
     *
     * Lets a caller dispatch on a tag value with no object in hand, recovering the payload
     * type inside @p visitor from the @c type_tag argument. Useful for size/type queries.
     *
     * @tparam Visitor  Callable accepting @c type_tag<payload> for the matching arm.
     * @tparam Fallback The return type; also the value returned when no arm matches.
     * @param tag      The enumerator to dispatch on.
     * @param visitor  The callable invoked with the matching @c type_tag.
     * @param fallback Value returned when @p tag matches no arm.
     * @return @c visitor(type_tag<payload>{}), or @p fallback when nothing matched.
     */
    template <typename Visitor, typename Fallback>
    ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE static Fallback
    visit_type(const Tag tag, Visitor&& visitor, Fallback fallback) noexcept {
        return visit_type_by_tag<Cases...>(tag, static_cast<Visitor&&>(visitor), fallback);
    }

private:
    /**
     * @brief Recursive implementation of @c visit_type: walks @c Cases to find @p tag.
     *
     * @tparam Case     The arm being tested this step.
     * @tparam Rest     The remaining arms.
     * @tparam Visitor  Callable accepting @c type_tag<payload>.
     * @tparam Fallback Return type and no-match value.
     * @param tag      The enumerator to match.
     * @param visitor  Callable invoked on a match.
     * @param fallback Returned once the arm list is exhausted with no match.
     * @return The visitor result, or @p fallback.
     */
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

    /**
     * @brief Recursive implementation of the mutable @c apply.
     *
     * Falls through silently (no fallback) when the tag matches no arm.
     *
     * @tparam Case    The arm being tested this step.
     * @tparam Rest    The remaining arms.
     * @tparam Visitor Callable accepting a mutable reference to the leaf.
     * @param owner   The object whose matched leaf is visited.
     * @param tag     The active tag driving the search.
     * @param visitor Callable invoked on the matched leaf.
     */
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

    /**
     * @brief Recursive implementation of @c construct: placement-news the matching payload.
     *
     * When the requested @p tag matches no arm, the recursion restarts from the full
     * @c Cases list at @c DefaultTag so a payload is always constructed. The
     * @c Case::tag != DefaultTag guard on the terminal step prevents infinite recursion:
     * once the default arm itself has failed to match, there is nothing left to retry.
     *
     * @tparam Case The arm being tested this step.
     * @tparam Rest The remaining arms.
     * @tparam Args Constructor argument types forwarded to the payload.
     * @param owner The object whose union member is constructed.
     * @param tag   The tag selecting the member.
     * @param args  Arguments forwarded to the payload constructor.
     */
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

    /**
     * @brief Recursive implementation of @c construct_payload: matches on payload type.
     *
     * Sets @c owner.type to the arm whose @c payload_type equals @p Payload and
     * copy-constructs it. If no arm matches, a dependent @c static_assert rejects the type
     * at compile time rather than silently doing nothing.
     *
     * @tparam Payload The concrete leaf type to store.
     * @tparam Case    The arm being tested this step.
     * @tparam Rest    The remaining arms.
     * @param owner   The object to initialize.
     * @param payload The value copied into the union.
     */
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

    /**
     * @brief Recursive implementation of @c copy_construct: copies the matching payload.
     *
     * Placement-news @p owner's member from the same member of @p other. If @p tag matches
     * no arm, the union is instead default-constructed at @c DefaultTag (guarded, as in
     * @c construct_by_tag, so the default arm does not retry itself).
     *
     * @tparam Case The arm being tested this step.
     * @tparam Rest The remaining arms.
     * @param owner The target object.
     * @param other The source object.
     * @param tag   The active tag driving the search.
     */
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

    /**
     * @brief Recursive implementation of @c destroy: destructs the matching payload.
     *
     * Explicitly calls the payload destructor via its type alias. No-ops if @p tag matches
     * no arm (nothing to destroy).
     *
     * @tparam Case The arm being tested this step.
     * @tparam Rest The remaining arms.
     * @param owner The object whose active member is destroyed.
     * @param tag   The active tag identifying the member.
     */
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

/**
 * @brief One arm of a @c DeviceTypeSwitch: pairs a tag with a payload type only.
 *
 * Unlike @c DeviceVariantCase there is no pointer-to-member and no storage; this maps tags
 * to types for pure type-level dispatch (no object involved).
 *
 * @tparam Tag      The enum type.
 * @tparam TagValue The enumerator this arm represents.
 * @tparam Payload  The type associated with @p TagValue.
 */
template <typename Tag, Tag TagValue, typename Payload>
struct DeviceTypeCase final {
    using payload_type = Payload; ///< Type associated with this arm's tag.

    static constexpr Tag tag = TagValue; ///< Enumerator this arm represents.
};

/**
 * @brief Stateless tag<->type mapping with no backing storage.
 *
 * A lightweight companion to @c DeviceVariant for cases that only need to translate
 * between an enumerator and a type: look up a type from a runtime tag (@c visit), test
 * whether a type is representable (@c holds), or recover the tag of a known type
 * (@c tag_of). It never touches or requires an object.
 *
 * @tparam Tag        The enum type used as the switch selector.
 * @tparam DefaultTag Tag returned by @c tag_of when the queried type is not represented.
 * @tparam Cases      The @c DeviceTypeCase arms.
 */
template <typename Tag, Tag DefaultTag, typename... Cases>
struct DeviceTypeSwitch final {
    /**
     * @brief Dispatches on a runtime @p tag, passing the matching type as a @c type_tag.
     *
     * @tparam Visitor  Callable accepting @c type_tag<payload> for the matched arm.
     * @tparam Fallback Return type and no-match value.
     * @param tag      The enumerator to dispatch on.
     * @param visitor  Callable invoked with the matched @c type_tag.
     * @param fallback Returned when @p tag matches no arm.
     * @return The visitor result, or @p fallback.
     */
    template <typename Visitor, typename Fallback>
    ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE static Fallback
    visit(const Tag tag, Visitor&& visitor, Fallback fallback) noexcept {
        return visit_by_tag<Cases...>(tag, static_cast<Visitor&&>(visitor), fallback);
    }

    /**
     * @brief Compile-time predicate: @c true iff @p Payload is one of the arms' types.
     *
     * @tparam Payload The type being tested for membership.
     */
    template <typename Payload>
    static constexpr bool holds = (std::is_same_v<Payload, typename Cases::payload_type> || ...);

    /**
     * @brief Returns the tag associated with @p Payload, or @c DefaultTag if unrepresented.
     *
     * @tparam Payload The type whose tag is requested.
     * @return The matching arm's tag, or @c DefaultTag when @p Payload is not present.
     */
    template <typename Payload>
    ATLAS_ALL_DEVICE ATLAS_FORCE_INLINE static constexpr Tag
    tag_of() noexcept {
        return tag_of_impl<Payload, Cases...>();
    }

private:
    /**
     * @brief Recursive implementation of @c tag_of.
     *
     * @tparam Payload The type being looked up.
     * @tparam Case    The arm being tested this step.
     * @tparam Rest    The remaining arms.
     * @return @c Case::tag on a type match, recurses otherwise, @c DefaultTag when exhausted.
     */
    template <typename Payload, typename Case, typename... Rest>
    ATLAS_ALL_DEVICE static constexpr Tag
    tag_of_impl() noexcept {
        if constexpr (std::is_same_v<Payload, typename Case::payload_type>) {
            return Case::tag;
        } else if constexpr (sizeof...(Rest) > 0) {
            return tag_of_impl<Payload, Rest...>();
        } else {
            return DefaultTag;
        }
    }

    /**
     * @brief Recursive implementation of @c visit.
     *
     * @tparam Case     The arm being tested this step.
     * @tparam Rest     The remaining arms.
     * @tparam Visitor  Callable accepting @c type_tag<payload>.
     * @tparam Fallback Return type and no-match value.
     * @param tag      The enumerator to match.
     * @param visitor  Callable invoked on a match.
     * @param fallback Returned when the arm list is exhausted with no match.
     * @return The visitor result, or @p fallback.
     */
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

} // namespace atlas