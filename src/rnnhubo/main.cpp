#include <stdio.h>
#include <string.h>
#include <string>
//#include <unordered_map>
#include <algorithm>

#include "rnn.h"

using namespace std;
//using std::unordered_map;
void InitHiddenInput(float *A,int dim,float value)
{
	for(int i=0;i<dim;++i)
		A[i]=value;
}
int main(int argc,char *argv[])
{
	if(argc != 5)
	{
		fprintf(stderr,"%s rnnfile inputlist outputlist nbestfile\n",argv[0]);
		return -1;
	}
	char *rnnfile = argv[1];
	Rnn rnn;
	rnn.LoadRNNLM(rnnfile);
	rnn.ReadWordlist(argv[2],argv[3]);
	
	char line[1024];
	char word[128];
	int word_id=0;
	memset(line,0x00,sizeof(line));
	memset(word,0x00,sizeof(word));
	char *nbestfile = argv[4];
	FILE *fp = fopen(nbestfile,"r");
	if(fp == NULL)
	{
		fprintf(stderr,"fopne %s error!\n",nbestfile);
		return -1;
	}
	int ioputdim = rnn.GetMaxLayerNode();
	int classdim = rnn.GetNclass();
	int hiddim = rnn.GetHidDim();
	float *rnninput;
	float *rnnoutput;
	float *classoutput;
	float *hidin = (float *)malloc(sizeof(float)*hiddim);
	if(hidin == NULL)
	{
		fprintf(stderr,"malloc error!\n");
		return -1;
	}
	if(classdim  > 0)
	{
		classoutput = (float *)malloc(sizeof(float)*classdim);
		if(classoutput == NULL)
		{
			fprintf(stderr,"malloc error!\n");
			return -1;
		}
	}
	rnninput = (float *)malloc(sizeof(float)*ioputdim);
	rnnoutput = (float *)malloc(sizeof(float)*ioputdim);
	if(rnninput == NULL || rnnoutput == NULL)
	{
		fprintf(stderr,"malloc error!\n");
		return -1;
	}
	while(fgets(line,sizeof(line),fp) != NULL)
	{
		InitHiddenInput(hidin,hiddim,0.1);
		line[strlen(line)-1] = '\0';
		char *p=NULL;
		char *cur_p=NULL;
		float amscore=0;
		float lmscore=0;
		int wordnum=0;
		int n=0;
		string word;
		char *cut_line = line;
		int curword=-1,prevword=-1;
		//memset(classoutput,0x00,sizeof(float)*classdim);
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
				if(strcmp(cur_p,"<s>") == 0)
				{
					prevword = curword;
					curword = rnn.GetStartIndex();
					continue;
				}
				else if(strcmp(cur_p,"</s>") == 0)
				{
					prevword=curword;
					curword = rnn.GetEndIndex();
				}
				else
				{
					prevword=curword;
					curword = rnn.GetWordId(word);
				}
				float rnnscore = rnn.forword(prevword,curword,rnninput,rnnoutput,ioputdim,
						hidin,hiddim,classoutput,classdim);
				printf("%f %s %d\n",rnnscore,cur_p,curword);
//				unordered_map<string,int>::iterator find_iter = word_list.find(word);
//				if(find_iter == word_list.end())//no search
//				{
//					fprintf(stderr,"find word %s error\n",word.c_str());
//					return -1;
//				}
//				int wordid = find_iter->second;
			}
			++n;
			cut_line = NULL;
		}
	}
	fclose(fp);
	free(rnninput);
	free(rnnoutput);
	free(classoutput);
	return 0;
}
