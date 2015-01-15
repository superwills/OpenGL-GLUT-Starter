#ifndef TIMER_H
#define TIMER_H

#ifdef _WIN32
#include <Windows.h>
#else
#include <sys/time.h>
#endif

// TIMER class, only available if C++ available
class Timer
{
  #ifdef _WIN32
  LARGE_INTEGER startTime ;
  double fFreq ;
  #else
  timeval startTime ;
  #endif
  
public:
  Timer() {
    #ifdef _WIN32
    LARGE_INTEGER freq ;
    QueryPerformanceFrequency( &freq ) ;
    fFreq = (double)freq.QuadPart ;
    #else
    gettimeofday( &startTime, NULL ) ;
    #endif
    reset();
  }

  void reset() {
    #ifdef _WIN32
    QueryPerformanceCounter( &startTime ) ;
    #else
    gettimeofday( &startTime, NULL ) ;
    #endif
  }

  // Gets the most up to date time, counting from the start time.
  // For this to be frame time, you must call RESET every frame.
  double getTime() const {
    #ifdef _WIN32
    LARGE_INTEGER endTime ;
    QueryPerformanceCounter( &endTime ) ;
    return ( endTime.QuadPart - startTime.QuadPart ) / fFreq ; // as double
    #else
    timeval endTime ;
    gettimeofday( &endTime, NULL ) ;
    return (endTime.tv_sec - startTime.tv_sec) + (endTime.tv_usec - startTime.tv_usec)/1e6 ;
    #endif
  }
} ;

#endif