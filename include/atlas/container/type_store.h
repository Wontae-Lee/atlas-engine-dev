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

template <typename Base>
class TypeStore final {
public:
    using key_type       = std::type_index;
    using mapped_type    = std::unique_ptr<Base>;
    using storage_type   = std::unordered_map<key_type, mapped_type>;
    using iterator       = typename storage_type::iterator;
    using const_iterator = typename storage_type::const_iterator;

    TypeStore()                     = default;
    TypeStore(const TypeStore&)     = delete;
    TypeStore(TypeStore&&) noexcept = default;
    ~TypeStore()                    = default;

    TypeStore&
    operator=(const TypeStore&)
        = delete;

    TypeStore&
    operator=(TypeStore&&) noexcept = default;

    ATLAS_HOST ATLAS_FORCE_INLINE void
    reserve(const std::size_t capacity) {
        _values.reserve(capacity);
    }

    ATLAS_NODISCARD ATLAS_HOST ATLAS_FORCE_INLINE std::size_t
    size() const noexcept {
        return _values.size();
    }

    ATLAS_NODISCARD ATLAS_HOST ATLAS_FORCE_INLINE bool
    empty() const noexcept {
        return _values.empty();
    }

    ATLAS_HOST ATLAS_FORCE_INLINE void
    clear() noexcept {
        _values.clear();
    }

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

    template <typename Value>
    ATLAS_NODISCARD ATLAS_HOST ATLAS_FORCE_INLINE Value*
    get() noexcept {
        static_assert(std::is_base_of_v<Base, Value>,
                      "Value must derive from the TypeStore base type.");

        const auto it = _values.find(key<Value>());
        return it == _values.end() ? nullptr : static_cast<Value*>(it->second.get());
    }

    template <typename Value>
    ATLAS_NODISCARD ATLAS_HOST ATLAS_FORCE_INLINE const Value*
    get() const noexcept {
        static_assert(std::is_base_of_v<Base, Value>,
                      "Value must derive from the TypeStore base type.");

        const auto it = _values.find(key<Value>());
        return it == _values.end() ? nullptr : static_cast<const Value*>(it->second.get());
    }

    template <typename Value>
    ATLAS_NODISCARD ATLAS_HOST ATLAS_FORCE_INLINE bool
    contains() const noexcept {
        static_assert(std::is_base_of_v<Base, Value>,
                      "Value must derive from the TypeStore base type.");
        return _values.contains(key<Value>());
    }

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

    ATLAS_NODISCARD ATLAS_HOST ATLAS_FORCE_INLINE iterator
    begin() noexcept {
        return _values.begin();
    }

    ATLAS_NODISCARD ATLAS_HOST ATLAS_FORCE_INLINE iterator
    end() noexcept {
        return _values.end();
    }

    ATLAS_NODISCARD ATLAS_HOST ATLAS_FORCE_INLINE const_iterator
    begin() const noexcept {
        return _values.begin();
    }

    ATLAS_NODISCARD ATLAS_HOST ATLAS_FORCE_INLINE const_iterator
    end() const noexcept {
        return _values.end();
    }

    ATLAS_NODISCARD ATLAS_HOST ATLAS_FORCE_INLINE const_iterator
    cbegin() const noexcept {
        return _values.cbegin();
    }

    ATLAS_NODISCARD ATLAS_HOST ATLAS_FORCE_INLINE const_iterator
    cend() const noexcept {
        return _values.cend();
    }

private:
    template <typename Value>
    ATLAS_NODISCARD ATLAS_HOST ATLAS_FORCE_INLINE static const key_type&
    key() noexcept {
        static_assert(std::is_base_of_v<Base, Value>,
                      "Value must derive from the TypeStore base type.");
        static const key_type key_value { typeid(Value) };
        return key_value;
    }

    storage_type _values;
};

}
