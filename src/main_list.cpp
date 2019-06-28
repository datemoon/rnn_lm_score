#include <stdio.h>
#include <string.h>
#include <string>
#include <unordered_map>
#include <algorithm>
#include <vector>
#include <sys/time.h>
#include "front_tree.h"

using namespace std;
using std::unordered_map;

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

	RnnCalc *rnncalc = new RnnCalc(rnn);
	CFsmLM *lm = new CFsmLM();
	lm->LoadLM(ngramemodel);
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
		FrontTreeClass front_tree(rnncalc, lm, ngramword_map_rnnword,
				start,end,0,NULL,NULL,NULL);
		fp = fopen(nbestfile,"r");
		if(fp == NULL)
		{
			fprintf(stderr,"fopen %s error!\n",nbestfile);
			return -1;
		}
		//create front_tree
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
					unordered_map<string,int>::iterator find_iter = word_list.find(word);
					if(find_iter == word_list.end())//no search
					{
						fprintf(stderr,"find word %s error\n",word.c_str());
						return -1;
					}
					int wordid = find_iter->second;
					if(wordid == end)
						front_tree.FindOrAddFrondTree(wordid,amscore);
					else
						front_tree.FindOrAddFrondTree(wordid);
				}
				++n;
				cut_line = NULL;
			}
		}
#ifdef DEBUG
		fprintf(stderr,"create tree end!\n");
#endif
		//create tree ok
		//start score
		front_tree.CalcScore(0.5,14);
		vector<int> path;

		front_tree.GetBestPath(path);
		printf("%s ",nbestfile);
		for(int i=0;i<path.size();++i)
		{
			printf("%s ",list_word[path[i]].c_str());
			//printf("%d ",path[i]);
		}
		printf("\n");
		//end score
		fclose(fp);
		front_tree.ClearFrondTree();
	}
	gettimeofday(&endtime,NULL);
	fprintf(stdout,"time is %f\n",endtime.tv_sec-starttime.tv_sec+(endtime.tv_usec-starttime.tv_usec)*1.0/1000000);
	fclose(fp_list);
	delete rnn;
	delete rnncalc;
	delete lm;
//	free(ngramword_map_rnnword);
	return 0;
}


