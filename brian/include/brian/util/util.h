#pragma once

namespace bb {
    template<typename T>
    class Optional {
    private:
        const bool m_hasValue;
        const T m_value;
    public:
        Optional(): m_value(NULL), m_hasValue(false) { }
        Optional(T&& p_value): m_value(p_value), m_hasValue(true) { }
        Optional(const Option<T>& p_original): m_value(p_original.m_value), m_hasValue(p_original.m_hasValue) { }

        inline bool has() {
            return this->m_hasValue;
        }

        inline T value() {
            return this->m_value;
        }
    };
}