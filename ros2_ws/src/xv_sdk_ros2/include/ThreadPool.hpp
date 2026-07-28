#include <iostream>
#include <vector>
#include <thread>
#include <queue>
#include <functional>
#include <mutex>
#include <condition_variable>

class ThreadPool
{
public:
    explicit ThreadPool(size_t nThreads) : m_stop(false)
    {
        for (size_t i = 0; i < nThreads; ++i)
        {
            m_workers.emplace_back([this]
                                   {
                for (;;) {
                    std::function<void()> task;

                    {
                        std::unique_lock<std::mutex> lock(this->m_mtx);
                        this->m_cv.wait(lock, [this] { return this->m_stop || !this->m_tasks.empty(); });
                        if (this->m_stop && this->m_tasks.empty()) return;

                        task = std::move(this->m_tasks.front());
                        this->m_tasks.pop();
                    }

                    task();
                } });
        }
    }

    template <class F, class... Args>
    void enqueue(F &&f, Args &&...args)
    {
        {
            std::unique_lock<std::mutex> lock(m_mtx);
            if (m_stop)
                throw std::runtime_error("enqueue on stopped ThreadPool");

            m_tasks.emplace(std::bind(std::forward<F>(f), std::forward<Args>(args)...));
        }
        m_cv.notify_one();
    }

    ~ThreadPool()
    {
        {
            std::unique_lock<std::mutex> lock(m_mtx);
            m_stop = true;
        }
        m_cv.notify_all();

        for (auto &worker : m_workers)
        {
            if (worker.joinable())
                worker.join();
        }
    }

private:
    std::vector<std::thread> m_workers;
    std::queue<std::function<void()>> m_tasks;

    std::mutex m_mtx;
    std::condition_variable m_cv;
    bool m_stop;
};
