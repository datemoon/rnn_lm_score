#include "front_tree_api.h"

int main(int argc,char *argv[])
{
	if(argc != 6)
	{
		fprintf(stderr,"%s ngramwordlistfile rnninwordlist rnnoutwordlist rnnmodel ngramemodel\n",argv[0]);
		return -1;
	}

	char * wordlistfile = argv[1];
    char * rnninwordlist = argv[2];
    char * rnnoutwordlist = argv[3];
    char * rnnmodel = argv[4];
    char * ngramemodel = argv[5];

	if(0 != InitRnnSource(wordlistfile,rnninwordlist,rnnoutwordlist,rnnmodel,ngramemodel))
	{
		fprintf(stderr,"InitRnnSource failed\n");
		return -1;
	}

//	onebest_res();
	DestoryRnnSource();
	return 0;
}
