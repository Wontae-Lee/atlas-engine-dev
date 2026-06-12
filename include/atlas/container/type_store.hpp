#pragma once

#include <utility>

namespace atlas {

template <typename Base>
void
TypeStore<Base>::reserve(const std::size_t capacity) {
    _values.reserve(capacity);
}

template <typename Base>
std::size_t
TypeStore<Base>::size() const noexcept {
    return _values.size();
}

template <typename Base>
bool
TypeStore<Base>::empty() const noexcept {
    return _values.empty();
}

template <typename Base>
void
TypeStore<Base>::clear() noexcept {
    _values.clear();
}

template <typename Base>
template <typename Value>
const typename TypeStore<Base>::key_type&
TypeStore<Base>::key() noexcept {
    static_assert(std::is_base_of_v<Base, Value>,
                  "Value must derive from the TypeStore base type.");
    static const key_type key { typeid(Value) };
    return key;
}

template <typename Base>
template <typename Value, typename... Args>
Value&
TypeStore<Base>::emplace(Args&&... args) {
    static_assert(std::is_base_of_v<Base, Value>,
                  "Value must derive from the TypeStore base type.");

    auto value = std::make_unique<Value>(std::forward<Args>(args)...);
    auto* ptr  = value.get();
    _values.insert_or_assign(key<Value>(), std::move(value));
    return *ptr;
}

template <typename Base>
template <typename Value>
void
TypeStore<Base>::set(std::unique_ptr<Value> value) {
    static_assert(std::is_base_of_v<Base, Value>,
                  "Value must derive from the TypeStore base type.");

    if (value == nullptr) {
        throw std::invalid_argument("TypeStore::set failed: value must not be null.");
    }

    _values.insert_or_assign(key<Value>(), std::move(value));
}

template <typename Base>
template <typename Value>
Value*
TypeStore<Base>::get() noexcept {
    static_assert(std::is_base_of_v<Base, Value>,
                  "Value must derive from the TypeStore base type.");

    const auto it = _values.find(key<Value>());
    return it == _values.end() ? nullptr : static_cast<Value*>(it->second.get());
}

template <typename Base>
template <typename Value>
const Value*
TypeStore<Base>::get() const noexcept {
    static_assert(std::is_base_of_v<Base, Value>,
                  "Value must derive from the TypeStore base type.");

    const auto it = _values.find(key<Value>());
    return it == _values.end() ? nullptr : static_cast<const Value*>(it->second.get());
}

template <typename Base>
template <typename Value>
bool
TypeStore<Base>::contains() const noexcept {
    static_assert(std::is_base_of_v<Base, Value>,
                  "Value must derive from the TypeStore base type.");
    return _values.contains(key<Value>());
}

template <typename Base>
template <typename Value>
std::unique_ptr<Value>
TypeStore<Base>::remove() {
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

template <typename Base>
typename TypeStore<Base>::iterator
TypeStore<Base>::begin() noexcept {
    return _values.begin();
}

template <typename Base>
typename TypeStore<Base>::iterator
TypeStore<Base>::end() noexcept {
    return _values.end();
}

template <typename Base>
typename TypeStore<Base>::const_iterator
TypeStore<Base>::begin() const noexcept {
    return _values.begin();
}

template <typename Base>
typename TypeStore<Base>::const_iterator
TypeStore<Base>::end() const noexcept {
    return _values.end();
}

template <typename Base>
typename TypeStore<Base>::const_iterator
TypeStore<Base>::cbegin() const noexcept {
    return _values.cbegin();
}

template <typename Base>
typename TypeStore<Base>::const_iterator
TypeStore<Base>::cend() const noexcept {
    return _values.cend();
}

} // namespace atlas
