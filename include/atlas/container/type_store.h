#pragma once

#include <atlas/core/macros.h>

#include <cstddef>
#include <memory>
#include <stdexcept>
#include <type_traits>
#include <typeindex>
#include <unordered_map>

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
    reserve(std::size_t capacity);

    ATLAS_HOST ATLAS_NODISCARD ATLAS_FORCE_INLINE std::size_t
    size() const noexcept;

    ATLAS_HOST ATLAS_NODISCARD ATLAS_FORCE_INLINE bool
    empty() const noexcept;

    ATLAS_HOST ATLAS_FORCE_INLINE void
    clear() noexcept;

    template <typename Value, typename... Args>
    ATLAS_HOST ATLAS_FORCE_INLINE Value&
    emplace(Args&&... args);

    template <typename Value>
    ATLAS_HOST ATLAS_FORCE_INLINE void
    set(std::unique_ptr<Value> value);

    template <typename Value>
    ATLAS_HOST ATLAS_NODISCARD ATLAS_FORCE_INLINE Value*
    get() noexcept;

    template <typename Value>
    ATLAS_HOST ATLAS_NODISCARD ATLAS_FORCE_INLINE const Value*
    get() const noexcept;

    template <typename Value>
    ATLAS_HOST ATLAS_NODISCARD ATLAS_FORCE_INLINE bool
    contains() const noexcept;

    template <typename Value>
    ATLAS_HOST ATLAS_NODISCARD ATLAS_FORCE_INLINE std::unique_ptr<Value>
    remove();

    ATLAS_HOST ATLAS_NODISCARD ATLAS_FORCE_INLINE iterator
    begin() noexcept;

    ATLAS_HOST ATLAS_NODISCARD ATLAS_FORCE_INLINE iterator
    end() noexcept;

    ATLAS_HOST ATLAS_NODISCARD ATLAS_FORCE_INLINE const_iterator
    begin() const noexcept;

    ATLAS_HOST ATLAS_NODISCARD ATLAS_FORCE_INLINE const_iterator
    end() const noexcept;

    ATLAS_HOST ATLAS_NODISCARD ATLAS_FORCE_INLINE const_iterator
    cbegin() const noexcept;

    ATLAS_HOST ATLAS_NODISCARD ATLAS_FORCE_INLINE const_iterator
    cend() const noexcept;

private:
    template <typename Value>
    ATLAS_HOST ATLAS_NODISCARD ATLAS_FORCE_INLINE static const key_type&
    key() noexcept;

    storage_type _values;
};

}

#include <atlas/container/type_store.hpp>