#ifndef __PTHREAD_HB__
#define __PTHREAD_HB__
#include <pthread.h>


struct Tpool_work
{
	void* (*fun)(void*);//task function
	void *arg ;         //parameter
	struct Tpool_work *next;
};

struct Tpool
{
	int shutdown;
	int max_thr_num;
	pthread_t *thr_id;
	struct Tpool_work* queue_head;
	pthread_mutex_t queue_lock;
	pthread_cond_t queue_ready;
};


//create pthread pool
int tpool_create(int max_thr_num,void *arg);

//destory pthread pool
void tpool_destory();

//add word to pthread pool
int tpool_add_work(void *(*fun)(void*),void *arg);

#endif 
