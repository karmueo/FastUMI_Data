#pragma once

#include <chrono>
#include <vector>
#include <mutex>

class FpsCount
{
  std::mutex mtx;
  std::vector<std::chrono::system_clock::time_point> m_tics;
  long long m_total = 0;

public:

  long long total() {
      std::lock_guard<std::mutex> l(mtx);
      return m_total;
  }

  void tic()
  {
    std::lock_guard<std::mutex> l(mtx);
    m_tics.push_back( std::chrono::system_clock::now() );
    ++m_total;
    while( std::chrono::duration_cast<std::chrono::milliseconds>( m_tics.back() - m_tics.front() ).count() > 1100 ){
      m_tics.erase( m_tics.begin() );
    }
  }

  double fps()
  {
    std::lock_guard<std::mutex> l(mtx);
    double fps = 0;
    const size_t &size = m_tics.size();
    if( size > 2 ){
      auto microseconds = std::chrono::duration_cast<std::chrono::microseconds>( m_tics.back() - m_tics.front() ).count();
      fps = 1000000.0 * static_cast<double>(size - 1) / microseconds;
    }
    return fps;
  }

  void reset()
  {
      std::lock_guard<std::mutex> l(mtx);
      m_tics.clear();
      m_total = 0;
  }
};
