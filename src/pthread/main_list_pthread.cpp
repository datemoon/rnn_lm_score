#include <stdio.h>
#include <string.h>
#include <string>
#include <unordered_map>
#include <algorithm>
#include <vector>
#include <sys/time.h>
#include "front_tree.h"
#include "pthread_hb.h"

using namespace std;
using std::unordered_map;
static pthread_mutex_t mutex_out = PTHREAD_MUTEX_INITIALIZER;

struct Args
{
	struct RnnAndNgraModel *model;
	char file[256];
	unordered_map<string,int> * word_list;
	vector<string> *list_word;
	Args(RnnAndNgraModel *_model,unordered_map<string,int> * _word_list,
			vector<string> *_list_word, char *_file):
		model(_model),word_list(_word_list),list_word(_list_word)
	{
		memset(file,0x00,sizeof(file));
		memcpy(file,_file,strlen(_file));
	}
};

void *fun(void *arg)
{
	struct Args *para = (struct Args *)arg ;

	RnnCalc *rnncalc = new RnnCalc(para->model->rnn);//this can be new at create pthread,but I didn't do it,now.
	FrontTreeClass front_tree(rnncalc,para->model->lm,para->model->ngramword_map_rnnword,
			para->model->start,para->model->end,
			0,NULL,NULL,NULL);
	FILE *fp = fopen(para->file,"r");
	if(fp == NULL)
	{
		fprintf(stderr,"fopen %s error!\n",para->file);
		return NULL;
	}
	char line[512];
	while(fgets(line,sizeof(line),fp) != NULL)
	{
		line[strlen(line)-1] = '\0';
		char *p=NULL;
		char *cur_p=NULL;
		float amscore=0;
		float lmscore=0;
		int wordnum=0;
		int n=0;
		string word;
		char *cut_line = line;
		while(NULL != (cur_p = strtok_r(cut_line," ",&p)))
		{
			if(n == 0)
				amscore = atof(cur_p);
			else if(n == 1)
				lmscore = atof(cur_p);
			else if(n == 2)
				wordnum = atoi(cur_p);
			else
			{
				word = cur_p;
				unordered_map<string,int>::iterator find_iter = para->word_list->find(word);
				if(find_iter == para->word_list->end())//no search
				{
					fprintf(stderr,"find word %s error\n",word.c_str());
					return NULL;
				}
				int wordid = find_iter->second;
				if(wordid == para->model->end)
					front_tree.FindOrAddFrondTree(wordid,amscore);
				else
					front_tree.FindOrAddFrondTree(wordid);
			}
			++n;
			cut_line = NULL;
		}
	}
	//create tree ok
	//start score
	front_tree.CalcScore(0.5,14,0.0);
	vector<int> path;
	front_tree.GetBestPath(path);
	pthread_mutex_lock(&mutex_out);
	printf("%s ",para->file);
	for(int i=0;i<path.size();++i)
	{
		printf("%s ",(*para->list_word)[path[i]].c_str());
	}
	printf("\n");
	fflush(stdout);
	pthread_mutex_unlock(&mutex_out);
	fclose(fp);
	front_tree.ClearFrondTree();

	delete rnncalc;
	delete para;
	return NULL;
}

int main(int argc,char *argv[])
{
	if(argc != 7)
	{
		fprintf(stderr,"%s ngramwordlistfile rnninwordlist rnnoutwordlist rnnmodel ngramemodel nbestfile\n",argv[0]);
		return -1;
	}
	char * wordlistfile = argv[1];
	char * rnninwordlist = argv[2];
	char * rnnoutwordlist = argv[3];
	char * rnnmodel = argv[4];
	char * ngramemodel = argv[5];
	char * nbestfilelist = argv[6];
	//because wordlist is different,so create map.
	unordered_map<string, int> word_list;//main list
	unordered_map<string, int> rnnword_list;
	vector<int> ngramword_map_rnnword;
	vector<string> list_word;
	int map_len=0;
	char line[1024];
	char word[128];
	int word_id=0;
	memset(line,0x00,sizeof(line));
	memset(word,0x00,sizeof(word));
	FILE *fp = fopen(rnninwordlist,"r");
	if(fp == NULL)
	{
		fprintf(stderr,"fopne %s error!\n",rnninwordlist);
		return -1;
	}
	while(fgets(line,sizeof(line),fp) != NULL)
	{
		sscanf(line,"%d %s",&word_id,word);
		rnnword_list[word] = word_id;
		memset(line,0x00,sizeof(line));
		memset(word,0x00,sizeof(word));
	}
	fclose(fp);
	fp = fopen(wordlistfile,"r");
	if(fp == NULL)
	{
		fprintf(stderr,"fopne %s error!\n",wordlistfile);
		return -1;
	}
	while(fgets(line,sizeof(line),fp) != NULL)
		++map_len;
	ngramword_map_rnnword.resize(map_len);
	list_word.resize(map_len);
/*	if(NULL == ngramword_map_rnnword)
	{
		fprintf(stderr,"malloc fail!\n");
		return -1;
	}
	memset(ngramword_map_rnnword,0x00,sizeof(int)*map_len);
*/	fseek(fp,0,SEEK_SET);
	while(fgets(line,sizeof(line),fp) != NULL)
	{
		sscanf(line,"%s %d",word,&word_id);
		word_list[word] = word_id;
		list_word[word_id] = word;
		unordered_map<string,int>::iterator find_iter =rnnword_list.find(word);
		if(strcmp("<s>",word) == 0 || strcmp("</s>",word) == 0)
		{
			ngramword_map_rnnword[word_id] = rnnword_list["<s>"];
		}
		else if(find_iter == word_list.end())
		{
			ngramword_map_rnnword[word_id] = rnnword_list["<OOS>"];
		}
		else
		{
			ngramword_map_rnnword[word_id] = find_iter->second;
		}
		memset(line,0x00,sizeof(line));
		memset(word,0x00,sizeof(word));
	}
	fclose(fp);
	//map ok
	int start = word_list["<s>"];
	int end = word_list["</s>"];
	Rnn *rnn = new Rnn();
	rnn->LoadRNNLM(rnnmodel);
	rnn->ReadWordlist(rnninwordlist,rnnoutwordlist);

	CFsmLM *lm = new CFsmLM();
	lm->LoadLM(ngramemodel);

	struct RnnAndNgraModel model(rnn,lm,ngramword_map_rnnword,start,end);

	int nthread = 4;
	printf("init thread\n");
	if(tpool_create(nthread,NULL) != 0)
	{
		fprintf(stderr,"tpool_create failed\n");
		exit(1);
	}

	struct timeval starttime,endtime;
	gettimeofday(&starttime,NULL);
	FILE *fp_list = fopen(nbestfilelist,"r");
	if(fp_list == NULL)
	{
		fprintf(stderr,"fopen %s error!\n",nbestfilelist);
		return -1;
	}
	char nbestfile[256];
	memset(nbestfile,0x00,sizeof(nbestfile));
	
	while(fgets(nbestfile,sizeof(nbestfile),fp_list) != NULL)
	{
		nbestfile[strlen(nbestfile)-1] = '\0';
		Args *work = new Args(&model,&word_list,&list_word,nbestfile);
		tpool_add_work(fun,(void*)work);
	}
	tpool_destory();
	gettimeofday(&endtime,NULL);
	fprintf(stdout,"time is %f\n",endtime.tv_sec-starttime.tv_sec+(endtime.tv_usec-starttime.tv_usec)*1.0/1000000);
	fclose(fp_list);

	delete rnn;
	delete lm;
//	free(ngramword_map_rnnword);
	return 0;
}


