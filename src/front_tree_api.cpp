#include "front_tree.h"
#include "front_tree_api.h"
#include <string.h>
static struct RnnAndNgraModel* model = NULL;

	vector<string> list_word;
	unordered_map<string, int> word_list;//main list
int InitRnnSource(char * wordlistfile,char * rnninwordlist,char * rnnoutwordlist,char * rnnmodel,
		char * ngramemodel)
{
//	unordered_map<string, int> word_list;//main list
	unordered_map<string, int> rnnword_list;
	vector<int> ngramword_map_rnnword;
//	vector<string> list_word;
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

	fseek(fp,0,SEEK_SET);
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

	model = new RnnAndNgraModel(rnn,lm,ngramword_map_rnnword,start,end);

	return 0;
}

int NbestRes(struct rec_NBEST_t *nbest,int n)
{
	if(nbest->nbest_num <= 0)
		return 0;

	RnnCalc *rnncalc = new RnnCalc(model->rnn);//this can be new at create pthread,but I didn't do it,now.

	FrontTreeClass front_tree(rnncalc,model->lm,model->ngramword_map_rnnword,
			model->start,model->end,
			0,NULL,NULL,NULL);
	int start = front_tree.GetStart();
	int end = front_tree.GetEnd();
	int i=0;
	for(i=0;i<nbest->nbest_num;++i)
	{
		int j=0;
		struct rec_1best_t *cur_res = nbest->nbest_rec[i];
		int wordnum = cur_res->wordnum;
		//int eps_flag = 0;
		for(j=0; j < wordnum;++j)
		{
			//start must be <s>,and only one;end it must be </s>, and only one.
			cur_res->words[j].wordid = word_list[cur_res->words[j].szword];
		/*	// no this condition.
			if(cur_res->words[j].wordid == 0)//0 is <eps>
			{
				eps_flag = 1;
				continue;
			}
			if(eps_flag == 1)
			{
				if(j == 1 && start != cur_res->words[j].wordid)
					front_tree.FindOrAddFrondTree(start);
			}
			else
			*/
			if(j == 0 && start != cur_res->words[j].wordid)
				front_tree.FindOrAddFrondTree(start);
			if(j == wordnum-1)
			{
#ifdef DEBUG
				if(cur_res->fbestpath_acscr < 0)
				{
					fprintf(stderr,"amscore %f < 0\n",cur_res->fbestpath_acscr);
				}
#endif
				if(cur_res->words[j].wordid != end)
				{
					front_tree.FindOrAddFrondTree(cur_res->words[j].wordid);
					front_tree.FindOrAddFrondTree(end,cur_res->fbestpath_acscr,1);
				}
				else
				{
					front_tree.FindOrAddFrondTree(cur_res->words[j].wordid,
							cur_res->fbestpath_acscr,1);
				}
			}
			else
				front_tree.FindOrAddFrondTree(cur_res->words[j].wordid);
		}
	}
	//create tree ok
	//start score
	front_tree.CalcScore(0.5,14,0.0);

#ifdef DEBUG
	vector<int> path;
	front_tree.GetBestPath(path);
	printf("**********\n");
	for(int i=0;i<path.size();++i)
	{
		printf("%s ",list_word[path[i]].c_str());
	}
	printf("best score %f\n************\n",front_tree.GetBestScore());
#endif
	int best_num = front_tree.GetBestPath();

	/*
	 * reorder nbest.
	 * */
	int *A = new int[n];
	float *F = new float[n];
	memset(A,0x00,sizeof(A));
	memset(F,0x00,sizeof(F));
	int len = front_tree.GetNBestPath(A,n,F);
#ifdef DEBUG
	if(len != n)
	{
		fprintf(stderr,"have some same result ,so %d != %d\n",len,n);
	}
#endif
//	rec_1best_t* cpynbest[nbest->nbest_num];
	rec_1best_t** cpynbest = new rec_1best_t*[nbest->nbest_num];
	memcpy(cpynbest,nbest->nbest_rec,nbest->nbest_num*sizeof(rec_1best_t*));
	for(int m=0;m<len;++m)
	{/*swap diff front len nbest,sort nbest*/
		cpynbest[A[m]]->fbestpath = F[m];
		nbest->nbest_rec[m] = cpynbest[A[m]];
	}
	if(best_num != A[0])
	{
		fprintf(stdout,"have some bug , i need modefy. best_num %d != A[0] %d\n",
				best_num,A[0]);
		return -1;
	}
	front_tree.ClearFrondTree();
	delete[] A;
	delete[] F;
	delete[] cpynbest;
	delete rnncalc;
	return len;
}

void DestoryRnnSource()
{
	delete model;
	model = NULL;
}

