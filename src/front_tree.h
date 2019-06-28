#ifndef __FRONT_TREE_H__
#define __FRONT_TREE_H__

#include <stdio.h>
#include <string.h>
#include <unordered_map>
#include <vector>
#include "FsmLM.h"
#include "rnn.h"
typedef int Label;
#define FLAGS 1

using namespace std;
using std::unordered_map;

/*   prev_tree -> cur_tree -> front_tree
 *                   |
 *                   v
 *                next_tree
 *
 *
 * */
struct FrontTree
{
#ifdef DEBUG
public:
	FrontTree(string word,Label word_id,FrontTree *prev_tree,FrontTree *front_tree,
			FrontTree *next):
		word_(word),word_id_(word_id),prev_tree_(prev_tree),
		front_tree_(front_tree),next_(next){}
	string word_;
#endif
	Label word_id_;
//	struct Token tok_;
	FrontTree *prev_tree_;
	struct FrontTree *front_tree_;
	struct FrontTree *next_;
public:

	FrontTree(Label word_id,FrontTree *prev_tree,FrontTree *front_tree,
			FrontTree *next):
		word_id_(word_id),prev_tree_(prev_tree),
		front_tree_(front_tree),next_(next){}
	FrontTree *GetNext()
	{ return next_;}
	FrontTree *GetPrev()
	{ return prev_tree_;}
	FrontTree *GetFront()
	{ return front_tree_;}
	int GetKey()
	{ return word_id_;}
};

struct RnnHiddenOut
{
	float *rnn_prev_hidden_out;//record rnn prev hidden out
	int  rnn_prev_hidden_out_dim;
};
struct RnnClassOut
{
	float *rnn_cur_class_out;
	int class_dim;
	int flag;//cur rnn class out whether calculate.if yes flag == 1,else flag == 0.
};

struct Token
{
//	int flag_;//mark whether have search front_tree,0 no,1 yes
	unsigned long long ngram_state_;
	float ngram_score_;//record ngrame score
	float rnn_score_;//record rnn score
	float total_score_;
	struct RnnHiddenOut rnn_hidden_out_;
	struct RnnClassOut rnn_class_out_;
//	Token * prev_; // no use
public:
	Token(int rnn_prev_hidden_out_dim,int rnn_class_out_dim):
		/*flag_(0),*/ngram_state_(0),ngram_score_(0),
		rnn_score_(0),total_score_(0)
	{
		rnn_hidden_out_.rnn_prev_hidden_out_dim = 
			rnn_prev_hidden_out_dim;
		rnn_hidden_out_.rnn_prev_hidden_out = 
			new float[rnn_prev_hidden_out_dim];
		rnn_class_out_.class_dim = rnn_class_out_dim;
		rnn_class_out_.rnn_cur_class_out =
			new float[rnn_class_out_dim];
		rnn_class_out_.flag = 0;
	}

	Token(Token* prev,int flag = 0):ngram_state_(prev->ngram_state_),ngram_score_(prev->ngram_state_),
	rnn_score_(prev->rnn_score_),total_score_(prev->total_score_)
	{
		rnn_hidden_out_.rnn_prev_hidden_out_dim = 
			prev->rnn_hidden_out_.rnn_prev_hidden_out_dim;
		rnn_hidden_out_.rnn_prev_hidden_out = 
			new float[rnn_hidden_out_.rnn_prev_hidden_out_dim];
		memcpy(rnn_hidden_out_.rnn_prev_hidden_out,
				prev->rnn_hidden_out_.rnn_prev_hidden_out,
				sizeof(float)*rnn_hidden_out_.rnn_prev_hidden_out_dim);
		rnn_class_out_.class_dim = prev->rnn_class_out_.class_dim;
		rnn_class_out_.rnn_cur_class_out =new float[rnn_class_out_.class_dim];
		rnn_class_out_.flag = flag;
		if(flag == 1)
		{
			memcpy(rnn_class_out_.rnn_cur_class_out,
					prev->rnn_class_out_.rnn_cur_class_out,
					sizeof(float)*rnn_class_out_.class_dim);
		}
	}

	~Token()
	{
		delete[] rnn_hidden_out_.rnn_prev_hidden_out;
		delete[] rnn_class_out_.rnn_cur_class_out;
	}
	void InitHidden(float value)
	{
		for(int i=0;i<rnn_hidden_out_.rnn_prev_hidden_out_dim;++i)
			rnn_hidden_out_.rnn_prev_hidden_out[i] = value;
	}
};

class FrontTreeClass
{
public:

	FrontTreeClass(RnnCalc *rnn, CFsmLM *lm, vector<int>&map,
			Label start, Label end,int node_num, FrontTree *tree_head, 
			FrontTree *cur_tree_link, FrontTree *prev_tree):
		rnn_(rnn),lm_(lm),
		start_wordid_(start),end_wordid_(end),node_num_(node_num),
		tree_head_(tree_head),cur_tree_link_(cur_tree_link),
		prev_tree_(prev_tree)
	{
#ifdef DEBUG
		leaf_num_ = 0;
		sentence_num_ = 0;
		calc_num_ = 0;
		nn = 0;
#endif
		ngramword_map_rnnword_ = map;
		maxleafscore_ = NULL;
		maxscore_ = -100000.0;
		lmrate_ = 0.5,lmweight_ = 14.0;
		cur_best_num_ = 0;
	}
	~FrontTreeClass()
	{
		ClearFrondTree();
		maxleafscore_ = NULL;
		start_wordid_=0,end_wordid_=0;
		node_num_=0,tree_head_=NULL;
		cur_tree_link_=NULL,prev_tree_=0;
		leaf.clear();
	}
	void FindOrAddFrondTree(Label word_id,float am_score=-1.0,int leaf_flag=0);
	void ClearFrondTree();
	void Reset()//every add new sentence
	{
		cur_tree_link_ = tree_head_;
	}
	FrontTree *GetTreeHead(){return tree_head_;}
	/*
	 * flag parameter is use for rnn speed up.
	 * */
	void Calc(Token *prev,FrontTree *curnode,int flag = 0);
	void CalcScore(float lmrate, float lmweight, float penalty = 0);
	float CalcNgramRnn(Token *tok, int word_id,int prev_word_id);

	inline bool IsStart(Label word_id)
	{
		if(word_id == start_wordid_)
			return true;
		return false;
	}
	inline Label GetStart(){ return start_wordid_;}

	inline Label GetEnd(){ return end_wordid_;}
	inline bool IsEnd(Label word_id)
	{
		if(word_id == end_wordid_)
			return true;
		return false;
	}
	void GetBestPath(vector<int> &path)
	{
		FrontTree *cur = maxleafscore_;
		while(cur != NULL)
		{
			path.push_back(cur->word_id_);
			cur = cur->prev_tree_;
		}
		//inverted order
		int size = path.size();
		for(int i=0;i<size/2;++i)
		{
			int tmp = path[i];
			path[i] = path[size-1-i];
			path[size-1-i] = tmp;
		}
#ifdef DEBUG
		fprintf(stderr,"best score %f\n",maxscore_);
#endif
	}

	int GetBestPath()
	{
		FrontTree *cur = maxleafscore_;
		return leaf[cur].best_num_;
	}

	int GetNBestPath(int *A,int len_A,float *F)
	{
		return ReorderLeafNode(A,len_A,F);
	}

	float GetBestScore(){return maxscore_;}
private:
	RnnCalc *rnn_ ;
	CFsmLM *lm_ ;
	FrontTree *maxleafscore_;
	float lmrate_;
	float lmweight_;
	float penalty_;
	float maxscore_;
	Label start_wordid_; //<s>
	Label end_wordid_;//</s>
	vector<int> ngramword_map_rnnword_;

#ifdef DEBUG
	int sentence_num_;
	int calc_num_;
	int leaf_num_;
	int nn;
#endif
	int max_socre_num_;
	struct LeafNode{
		float amscore_;
		int best_num_;
	};//for hawkdecode the best path
	int node_num_;
	FrontTree *tree_head_;
	FrontTree *cur_tree_link_;
	FrontTree *prev_tree_;
	//unordered_map<FrontTree *,float> leaf;
	unordered_map<FrontTree *,LeafNode> leaf;
	vector<FrontTree *> leaf_node;
	int cur_best_num_;
private:
	/*
	 * in leaf_node ,FrontTree * is sort by amscore_.
	 * */
	int ReorderLeafNode(int *A,int len_A,float *F);
};
//for thread 
struct RnnAndNgraModel
{
	Rnn *rnn;
	CFsmLM *lm;
	vector<int> ngramword_map_rnnword;
	int start;
	int end;
	RnnAndNgraModel(Rnn *_rnn,CFsmLM *_lm,vector<int>&_ngramword_map_rnnword,int _start,int _end):rnn(_rnn),lm(_lm),start(_start),end(_end)
	{
		ngramword_map_rnnword = _ngramword_map_rnnword;
	}
};

class SearchFrontTree
{
public:
	
private:
	FrontTree *tree_head_;
};
#endif
