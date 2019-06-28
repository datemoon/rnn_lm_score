#include <errno.h>
#include <unistd.h>
#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include "pthread_hb.h"
static struct Tpool *tpool = NULL;
/**
 *  args for pthread init ,at here isn't use.
 * */
static void * thread_fun(void *arg)
{
	struct Tpool_work *work;
	while(1)
	{
		pthread_mutex_lock(&tpool->queue_lock);
		while(!tpool->queue_head && !tpool->shutdown)
		{
			pthread_cond_wait(&tpool->queue_ready, &tpool->queue_lock);
		}
		if (tpool->shutdown)
		{
//			vad_destory_score(score_handle);
			pthread_mutex_unlock(&tpool->queue_lock);
			pthread_exit(NULL);
		}
		work = tpool->queue_head;
		tpool->queue_head = tpool->queue_head->next;
		pthread_mutex_unlock(&tpool->queue_lock);
		
		work->fun(work->arg);
		free(work);
	}
}

int tpool_create(int max_thr_num,void *arg)
{
	int i;
	tpool = (struct Tpool*)calloc(1, sizeof(struct Tpool));
	if (!tpool)
	{
		printf("%s: calloc failed\n", __FUNCTION__);
		exit(1);
	}
	tpool->max_thr_num = max_thr_num;
	tpool->shutdown = 0;
	tpool->queue_head = NULL;
	if (pthread_mutex_init(&tpool->queue_lock, NULL) !=0)
	{
		printf("%s: pthread_mutex_init failed, errno:%d, error:%s\n",
				__FUNCTION__, errno, strerror(errno));
		exit(1);
	}
	if (pthread_cond_init(&tpool->queue_ready, NULL) !=0 )
	{
		printf("%s: pthread_cond_init failed, errno:%d, error:%s\n", 
				__FUNCTION__, errno, strerror(errno));
		exit(1);
	}
	tpool->thr_id = (pthread_t*)calloc(max_thr_num, sizeof(pthread_t));
	if (!tpool->thr_id)
	{
		printf("%s: calloc failed\n", __FUNCTION__);
		exit(1);
	}
	for (i = 0; i < max_thr_num; ++i)
	{
		if (pthread_create(&tpool->thr_id[i], NULL, thread_fun, arg) != 0)
		{
			printf("%s:pthread_create failed, errno:%d, error:%s\n", __FUNCTION__, 
					errno, strerror(errno));
			exit(1);
		}
	}
	return 0;
}

void tpool_destory()
{
	int i=0;
	struct Tpool_work *member;

	if(tpool->shutdown)
		return ;

	//tpool->shutdown = 1;
	while(1)
	{
		pthread_mutex_lock(&tpool->queue_lock);
		if(tpool->queue_head == NULL)
		{
			tpool->shutdown = 1;
			pthread_mutex_unlock(&tpool->queue_lock);
			break;
		}
		pthread_mutex_unlock(&tpool->queue_lock);
		int seconds=1;
		//printf("sleep %d\n",seconds);
		sleep(seconds);
	}

	pthread_mutex_lock(&tpool->queue_lock);
	pthread_cond_broadcast(&tpool->queue_ready);
	pthread_mutex_unlock(&tpool->queue_lock);
	for (i = 0; i < tpool->max_thr_num; ++i)
	{
		pthread_join(tpool->thr_id[i], NULL);
	}
	free(tpool->thr_id);
	while(tpool->queue_head)
	{
		member = tpool->queue_head;
		tpool->queue_head = tpool->queue_head->next;
		free(member);
	}

	pthread_mutex_destroy(&tpool->queue_lock);
	pthread_cond_destroy(&tpool->queue_ready);
	free(tpool);
}

int tpool_add_work(void*(*fun)(void*),void *arg)
{
	struct Tpool_work *work,*member;

	if(!fun)
	{
		printf("%s:Invalid argument\n", __FUNCTION__);
		return -1;
	}

	work = (struct Tpool_work*)malloc(sizeof(struct Tpool_work));
	if(NULL == work)
	{
		printf("%s:malloc failed\n", __FUNCTION__);
		return -1;
	}
	work->fun = fun;
	work->arg = arg;
	work->next = NULL;

	pthread_mutex_lock(&tpool->queue_lock);    
	member = tpool->queue_head;
	if (!member)
	{
		tpool->queue_head = work;
	}
	else
	{
		while(member->next)
		{
			member = member->next;
		}
		member->next = work;
	}
	pthread_cond_signal(&tpool->queue_ready);
//	printf("send signal\n");
	pthread_mutex_unlock(&tpool->queue_lock);

	return 0;

}

#ifdef TEST
void *fun(void *arg)
{
	printf("thread %d\n",(int)arg);
	sleep(1);
	return NULL;
}

int main(int argc,char *argv[])
{
	if(tpool_create(3) != 0)
	{
		printf("tpool_create failed\n");
		exit(1);
	}

	int i=0;
	for (i = 0; i < 10; ++i)
	{
		tpool_add_work(fun, (void*)i);
	}
	//sleep(100);
	tpool_destory();
	return 0;
}

#endif
