#include <queue>
#include <mutex>

template <typename T>
class safe_queue
{
    mutable std::mutex m_mut;
    std::queue<T> m_data;
    long m_max_length;

public:
    safe_queue(long n) : m_max_length(n) {}

    void push(T new_value)
    {
        std::lock_guard<std::mutex> lk(m_mut);
        if (m_max_length > 0 && long(m_data.size()) >= m_max_length)
            m_data.pop();
        m_data.push(new_value);
    }

    T pop()
    {
        std::lock_guard<std::mutex> lk(m_mut);
        T t;
        if (!m_data.empty())
        {
            t = m_data.front();
            m_data.pop();
        }
        return t;
    }

    void clear()
    {
        std::lock_guard<std::mutex> lk(m_mut);
        std::queue<T>().swap(m_data);
    }

    bool empty() const
    {
        std::lock_guard<std::mutex> lk(m_mut);
        return m_data.empty();
    }
};
