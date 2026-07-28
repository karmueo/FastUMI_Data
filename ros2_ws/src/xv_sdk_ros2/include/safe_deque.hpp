#include <deque>
#include <mutex>

template <typename T>
class safe_deque
{
    mutable std::mutex m_mut;
    std::deque<T> m_data;
    long m_max_length;

public:
    safe_deque() : m_max_length(-1) {}
    safe_deque(long n) : m_max_length(n) {}

    void push(T new_value)
    {
        {
            std::lock_guard<std::mutex> lk(m_mut);

            if (m_max_length > 0 && long(m_data.size()) >= m_max_length)
                m_data.pop_front();
            m_data.push_back(new_value);
        }
    }

    bool try_pop(T &value)
    {
        {
            std::lock_guard<std::mutex> lk(m_mut);
            if (m_data.empty())
                return false;

            value = m_data.front();
            m_data.pop_front();
            return true;
        }
    }

    void clear()
    {
        {
            std::lock_guard<std::mutex> lk(m_mut);
            std::deque<T>().swap(m_data);
        }
    }

    bool empty() const
    {
        {
            std::lock_guard<std::mutex> lk(m_mut);
            return m_data.empty();
        }
    }

    int size() const
    {
        {
            std::lock_guard<std::mutex> lk(m_mut);
            return m_data.size();
        }
    }

    T back() const
    {
        {
            std::lock_guard<std::mutex> lk(m_mut);
            return m_data.back();
        }
    }
};
