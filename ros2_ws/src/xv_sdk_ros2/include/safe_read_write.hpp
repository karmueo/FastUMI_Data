#pragma once
#include <mutex>

template <class T>
class safe_rw
{
    mutable std::mutex m_mut;
    T m_data;

public:
    T read() const
    {
        std::lock_guard<std::mutex> guard(m_mut);
        T t = m_data;
        return t;
    }

    void write(T value)
    {
        std::lock_guard<std::mutex> guard(m_mut);
        m_data = value;
    }
};
