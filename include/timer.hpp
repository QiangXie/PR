#ifndef _TIMER_HPP_
#define _TIMER_HPP_
#include <sys/time.h>

class Timer_{
	public:
		inline void tic(){
			gettimeofday(&t1, NULL);
		}
		inline void toc(){
			gettimeofday(&t2, NULL);
		}
	        inline float getTotalTime(){
			float ms =  float((t2.tv_sec-t1.tv_sec) * 1000000 + t2.tv_usec-t1.tv_usec)/1000.0;
			return ms;
		}
	private:
		struct timeval t1, t2;
			
};

#endif
