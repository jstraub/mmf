#ifndef SIMPLE_PROFILER_H
#define SIMPLE_PROFILER_H
#include <map>
#include "time.h"
#include <sys/time.h>
#include <string>
#include <stdio.h>

using std::string;

using std::map;
class SimpleProfiler{
struct ProfileRecord{
    ProfileRecord(){
        usec=0;
        sec=0;
    }
    struct timeval start;
    long usec;
    long sec;
};
map<string,ProfileRecord> records;
static SimpleProfiler *_instance;
SimpleProfiler(){
}

public:
static SimpleProfiler *getInstance(){
if (!_instance){
    _instance=new SimpleProfiler();
}
return _instance;
}
void StartWatch(const string& name){
    gettimeofday(&(records[name].start), NULL);
}
void StopWatch(const string& name){
    timeval stop;
     gettimeofday(&stop, NULL);

     records[name].usec+=stop.tv_usec-records[name].start.tv_usec;
     records[name].sec+=stop.tv_sec-records[name].start.tv_sec;
     if (records[name].usec>1e6){
     unsigned long residual=records[name].usec%1000000;
     unsigned long sec=(records[name].usec-residual)/1000000;
  //   fprintf(stderr,"usec: %lu, residual: %lu, sec: %lu",records[name].usec,residual,sec);
     records[name].usec-=(sec*1000000);
     records[name].sec+=(sec);
     }
     if (records[name].usec<0){
         records[name].usec+=1e6;
         records[name].sec--;
     }
}
void PrintWatches(){
    for(map<string,ProfileRecord>::iterator iter = records.begin();iter!=records.end();iter++){
    fprintf(stderr,"%s: %ld sec, %ld usec\n",iter->first.c_str(),iter->second.sec,iter->second.usec);
    }
}
void PrintWatch(const string& name){
    fprintf(stderr,"%s: %ld sec, %ld usec\n",name.c_str(),records[name].sec,records[name].usec);
}
};
#endif
