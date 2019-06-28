#ifndef __FRONT_TREE_API_H__
#define __FRONT_TREE_API_H__ 1

#include "macro.h"

int InitRnnSource(char * wordlistfile,char * rnninwordlist,char * rnnoutwordlist,
		char * rnnmodel,char * ngramemodel);


int NbestRes(struct rec_NBEST_t *nbest,int n);

void DestoryRnnSource();

#endif
