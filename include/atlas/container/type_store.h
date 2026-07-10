#pragma once

#include <atlas/core/macros.h>

#include <cstddef>
#include <memory>
#include <stdexcept>
#include <type_traits>
#include <typeindex>
#include <unordered_map>
#include <utility>

namespace atlas {

/**
 * @brief A host-side heterogeneous container keyed by C++ static type.
 *
 * `TypeStore` maps each concrete derived type to at most one owned instance of it, using
 * the type itself as the key (`std::type_index` of `typeid(Value)`). It is a type-indexed
 * set of polymorphic singletons: there can be only one stored `Value` per distinct
 * `Value`, and inserting a second one replaces the first. Atlas uses it to hold the
 * per-simulation state objects that different subsystems attach — e.g.
 * `FluidStateStore = TypeStore<FluidState>` and
 * `UniverseStateStore = TypeStore<UniverseState>` — so a subsystem can stash and later
 * retrieve its own state type without a hand-maintained enum or registry.
 *
 * Values are owned through `std::unique_ptr<Base>` and always accessed as their concrete
 * type via `static_cast`; the cast is sound because a value is only ever retrieved under
 * the exact key it was stored with.
 *
 * @note Host-only. It relies on RTTI, the standard heap, and `std::unordered_map`, none of
 *       which are device-callable, so no member is annotated `__device__`.
 * @note Move-only: copying is deleted because `std::unique_ptr` elements are not copyable;
 *       moving transfers ownership of the whole map.
 * @warning Not thread-safe. Concurrent mutation and lookup must be externally synchronized.
 *
 * @tparam Base Common base class of every stored value. Each `Value` inserted must derive
 *              from `Base` (enforced by `static_assert`) and `Base` must have a virtual
 *              destructor for the owning `unique_ptr<Base>` to destroy derived objects
 *              correctly.
 */
template <typename Base>
class TypeStore final {
public:
    /// The map key: the `std::type_index` identifying a stored value's concrete type.
    using key_type       = std::type_index;
    /// The owned value type: a `unique_ptr` to the common base.
    using mapped_type    = std::unique_ptr<Base>;
    /// The underlying associative container from type key to owned value.
    using storage_type   = std::unordered_map<key_type, mapped_type>;
    /// Mutable iterator over the underlying map's `{key_type, mapped_type}` pairs.
    using iterator       = typename storage_type::iterator;
    /// Const iterator over the underlying map's `{key_type, mapped_type}` pairs.
    using const_iterator = typename storage_type::const_iterator;

    /// Constructs an empty store.
    TypeStore()                     = default;
    /// Copying is deleted: the owned `unique_ptr` values are not copyable.
    TypeStore(const TypeStore&)     = delete;
    /// Move-constructs by transferring ownership of the entire map.
    TypeStore(TypeStore&&) noexcept = default;
    /// Destroys the store and every value it owns (via each `unique_ptr<Base>`).
    ~TypeStore()                    = default;

    /// Copy assignment is deleted for the same reason copy construction is.
    TypeStore&
    operator=(const TypeStore&)
        = delete;

    /// Move-assigns by transferring ownership of the entire map.
    TypeStore&
    operator=(TypeStore&&) noexcept = default;

    /**
     * @brief Reserves space for at least @p capacity distinct value types.
     * @param capacity Expected number of entries; a hint to reduce rehashing.
     */
    ATLAS_HOST ATLAS_FORCE_INLINE void
    reserve(const std::size_t capacity) {
        _values.reserve(capacity);
    }

    /**
     * @brief Number of distinct value types currently held.
     * @return The count of stored entries.
     */
    ATLAS_NODISCARD ATLAS_HOST ATLAS_FORCE_INLINE std::size_t
    size() const noexcept {
        return _values.size();
    }

    /**
     * @brief Whether the store holds no values.
     * @return `true` iff nothing is stored.
     */
    ATLAS_NODISCARD ATLAS_HOST ATLAS_FORCE_INLINE bool
    empty() const noexcept {
        return _values.empty();
    }

    /**
     * @brief Destroys and removes every stored value, leaving the store empty.
     */
    ATLAS_HOST ATLAS_FORCE_INLINE void
    clear() noexcept {
        _values.clear();
    }

    /**
     * @brief Constructs a `Value` in place and stores it under its own type key.
     *
     * Allocates a new `Value` from the forwarded constructor arguments and inserts it under
     * `key<Value>()`. If a `Value` was already present it is destroyed and replaced
     * (`insert_or_assign`), preserving the one-instance-per-type invariant.
     *
     * @tparam Value Concrete type to construct; must derive from @p Base.
     * @tparam Args  Constructor argument types for @p Value.
     * @param args Arguments forwarded to `Value`'s constructor.
     * @return Reference to the newly stored `Value`, valid until it is replaced or removed.
     */
    template <typename Value, typename... Args>
    ATLAS_HOST ATLAS_FORCE_INLINE Value&
    emplace(Args&&... args) {
        static_assert(std::is_base_of_v<Base, Value>,
                      "Value must derive from the TypeStore base type.");

        auto value = std::make_unique<Value>(std::forward<Args>(args)...);
        auto* ptr  = value.get();
        _values.insert_or_assign(key<Value>(), std::move(value));
        return *ptr;
    }

    /**
     * @brief Stores an already-constructed value, taking ownership of it.
     *
     * Inserts @p value under `key<Value>()`, replacing any existing `Value`.
     *
     * @tparam Value Concrete type of the value; must derive from @p Base.
     * @param value Owning pointer to the value; ownership is transferred into the store.
     * @throws std::invalid_argument when @p value is null.
     */
    template <typename Value>
    ATLAS_HOST ATLAS_FORCE_INLINE void
    set(std::unique_ptr<Value> value) {
        static_assert(std::is_base_of_v<Base, Value>,
                      "Value must derive from the TypeStore base type.");

        if (value == nullptr) {
            throw std::invalid_argument("TypeStore::set failed: value must not be null.");
        }

        _values.insert_or_assign(key<Value>(), std::move(value));
    }

    /**
     * @brief Retrieves a mutable pointer to the stored `Value`, or null if absent.
     *
     * The returned pointer is non-owning; the store retains ownership. The `static_cast`
     * back to `Value*` is safe because the entry was stored under this exact type key.
     *
     * @tparam Value Concrete type to look up; must derive from @p Base.
     * @return Pointer to the stored `Value`, or `nullptr` if no `Value` is present.
     */
    template <typename Value>
    ATLAS_NODISCARD ATLAS_HOST ATLAS_FORCE_INLINE Value*
    get() noexcept {
        static_assert(std::is_base_of_v<Base, Value>,
                      "Value must derive from the TypeStore base type.");

        const auto it = _values.find(key<Value>());
        return it == _values.end() ? nullptr : static_cast<Value*>(it->second.get());
    }

    /**
     * @brief Retrieves a const pointer to the stored `Value`, or null if absent.
     * @tparam Value Concrete type to look up; must derive from @p Base.
     * @return Const pointer to the stored `Value`, or `nullptr` if no `Value` is present.
     */
    template <typename Value>
    ATLAS_NODISCARD ATLAS_HOST ATLAS_FORCE_INLINE const Value*
    get() const noexcept {
        static_assert(std::is_base_of_v<Base, Value>,
                      "Value must derive from the TypeStore base type.");

        const auto it = _values.find(key<Value>());
        return it == _values.end() ? nullptr : static_cast<const Value*>(it->second.get());
    }

    /**
     * @brief Tests whether a `Value` is currently stored.
     * @tparam Value Concrete type to test for; must derive from @p Base.
     * @return `true` iff a `Value` is present.
     */
    template <typename Value>
    ATLAS_NODISCARD ATLAS_HOST ATLAS_FORCE_INLINE bool
    contains() const noexcept {
        static_assert(std::is_base_of_v<Base, Value>,
                      "Value must derive from the TypeStore base type.");
        return _values.contains(key<Value>());
    }

    /**
     * @brief Detaches the stored `Value` and returns ownership of it to the caller.
     *
     * Releases the entry from its `unique_ptr<Base>` and re-wraps it as a
     * `unique_ptr<Value>` (safe cast, same reasoning as `get()`), then erases the map
     * entry so the store no longer references it.
     *
     * @tparam Value Concrete type to remove; must derive from @p Base.
     * @return Owning pointer to the removed `Value`, or `nullptr` if none was present.
     */
    template <typename Value>
    ATLAS_NODISCARD ATLAS_HOST ATLAS_FORCE_INLINE std::unique_ptr<Value>
    remove() {
        static_assert(std::is_base_of_v<Base, Value>,
                      "Value must derive from the TypeStore base type.");

        const auto it = _values.find(key<Value>());
        if (it == _values.end()) {
            return nullptr;
        }

        auto value = std::unique_ptr<Value>(static_cast<Value*>(it->second.release()));
        _values.erase(it);
        return value;
    }

    /**
     * @brief Mutable iterator to the first stored entry.
     *
     * Iteration yields `{key_type, mapped_type}` pairs in unspecified (hash) order; the
     * mapped value is a `unique_ptr<Base>`.
     *
     * @return Iterator to the beginning of the underlying map.
     */
    ATLAS_NODISCARD ATLAS_HOST ATLAS_FORCE_INLINE iterator
    begin() noexcept {
        return _values.begin();
    }

    /**
     * @brief Mutable past-the-end iterator.
     * @return Iterator one past the last stored entry.
     */
    ATLAS_NODISCARD ATLAS_HOST ATLAS_FORCE_INLINE iterator
    end() noexcept {
        return _values.end();
    }

    /**
     * @brief Const iterator to the first stored entry.
     * @return Const iterator to the beginning of the underlying map.
     */
    ATLAS_NODISCARD ATLAS_HOST ATLAS_FORCE_INLINE const_iterator
    begin() const noexcept {
        return _values.begin();
    }

    /**
     * @brief Const past-the-end iterator.
     * @return Const iterator one past the last stored entry.
     */
    ATLAS_NODISCARD ATLAS_HOST ATLAS_FORCE_INLINE const_iterator
    end() const noexcept {
        return _values.end();
    }

    /**
     * @brief Explicit const iterator to the first stored entry.
     * @return Const iterator to the beginning of the underlying map.
     */
    ATLAS_NODISCARD ATLAS_HOST ATLAS_FORCE_INLINE const_iterator
    cbegin() const noexcept {
        return _values.cbegin();
    }

    /**
     * @brief Explicit const past-the-end iterator.
     * @return Const iterator one past the last stored entry.
     */
    ATLAS_NODISCARD ATLAS_HOST ATLAS_FORCE_INLINE const_iterator
    cend() const noexcept {
        return _values.cend();
    }

private:
    /**
     * @brief Returns the stable type key for @p Value.
     *
     * The key is a function-local `static` `std::type_index` built once from
     * `typeid(Value)`, so every call for the same `Value` returns a reference to the same
     * object — this both amortizes construction and guarantees a consistent key across all
     * operations for a given type.
     *
     * @tparam Value Concrete type whose key is requested; must derive from @p Base.
     * @return Const reference to the cached `type_index` for @p Value.
     */
    template <typename Value>
    ATLAS_NODISCARD ATLAS_HOST ATLAS_FORCE_INLINE static const key_type&
    key() noexcept {
        static_assert(std::is_base_of_v<Base, Value>,
                      "Value must derive from the TypeStore base type.");
        static const key_type key_value { typeid(Value) };
        return key_value;
    }

    /// The type-keyed map owning one value per distinct stored type.
    storage_type _values;
};

}