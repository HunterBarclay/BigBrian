#pragma once

namespace bb {
    template<typename T>
    class Optional {
    private:
        const bool m_hasValue;
        const T m_value;
    public:
        Optional(): m_value(T()), m_hasValue(false) { }
        Optional(T value): m_value(value), m_hasValue(true) { }
        Optional(const Optional<T>& original): m_value(original.m_value), m_hasValue(original.m_hasValue) { }

        inline bool has() {
            return this->m_hasValue;
        }

        inline T value() {
            return this->m_value;
        }
    };

    template<typename T>
    Optional<T> available(T value) {
        return Optional<T>(value);
    }

    template<typename T>
    Optional<T> unavailable() {
        return Optional<T>();
    }
}