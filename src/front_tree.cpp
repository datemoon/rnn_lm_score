#include <stdio.h>
#include <math.h>
#include "front_tree.h"
//if add word_id isn't setence end ,am_score no use.
void FrontTreeClass::FindOrAddFrondTree(Label word_id,float am_score,int leaf_flag)
{
	FrontTree * tree = NULL;
	do{
		if(NULL == tree_head_)
		{
			tree = new FrontTree(word_id,NULL,NULL,NULL);
			tree_head_ = tree;
			++node_num_;
			cur_tree_link_ = tree_head_;
		//	prev_tree_link = NULL;
			break ;
		}
		if(word_id == start_wordid_ )
		{
			tree = tree_head_;
		   	cur_tree_link_ = tree_head_;
			break ;
		}
		prev_tree_ = cur_tree_link_;
	
		if(prev_tree_->front_tree_ == NULL)
		{
			tree = new FrontTree(word_id,prev_tree_,NULL,NULL);
			++node_num_;
			prev_tree_->front_tree_ = tree;
			cur_tree_link_ = tree;
		}
		else
		{
			int flag = 0;
			cur_tree_link_ = prev_tree_->front_tree_;
			while(cur_tree_link_ != NULL)//search wordid
			{
				tree = cur_tree_link_;
				if(cur_tree_link_->word_id_ == word_id)
				{
					flag = 1;
					break;
				}
				if(cur_tree_link_->next_ == NULL)
					break;
				cur_tree_link_ = cur_tree_link_->next_;
			}
			if(flag != 1)//no search wordid,add new node
			{
				tree = new FrontTree(word_id,prev_tree_,NULL,NULL);
				++node_num_;
				cur_tree_link_->next_ = tree;
				cur_tree_link_ = tree;
			}
		}
	}while(0);
	if(am_score > 0 || leaf_flag == 1)//because sometimes am_score < 0,so add leaf_flag for leaf node.
	{
		if(tree == NULL)
		{
			fprintf(stderr,"tree shouldn't NULL\n");
			return ;
		}
		if(leaf.find(tree) == leaf.end())//no search
		{
			leaf[tree].amscore_ = am_score;
			leaf[tree].best_num_ = cur_best_num_;
			leaf_node.push_back(tree);
#ifdef DEBUG
			++leaf_num_;
#endif
		}
		else
		{
#ifdef DEBUG
			fprintf(stdout,"find same sentence!\n");
#endif
			if(leaf[tree].amscore_ < am_score)
			{
				leaf[tree].amscore_ = am_score;
				leaf[tree].best_num_ = cur_best_num_;

			}
			//leaf[tree] = leaf[tree]>am_score ? leaf[tree] : am_score;
		}
		cur_best_num_++;
	}
	return ;
}

int FrontTreeClass::ReorderLeafNode(int *A,int len_A,float *F)
{
	/*
	 * bubble sort
	 * */
	int i=0,j=0;
	int len = 0;
	if(len_A > leaf_node.size())
		len = leaf_node.size();
	else
		len = len_A;
	//leaf_node sort
	int max = 0;
	for(i=0; i<len; ++i)
	{
		max = i;
		float maxscore = leaf[leaf_node[i]].amscore_;
		for(j=i+1;j<leaf_node.size(); ++j)
		{
			float curscore = leaf[leaf_node[j]].amscore_;
			if(maxscore < curscore)
			{
				FrontTree *tmp = leaf_node[j];
				leaf_node[j] = leaf_node[i];
				leaf_node[i] = tmp;
				maxscore = curscore;
			}
		}
	}
	for(i=0;i<len;++i)
	{
		A[i] = leaf[leaf_node[i]].best_num_;
		F[i] = leaf[leaf_node[i]].amscore_;
	}
	return len;
}

void FrontTreeClass::ClearFrondTree()
{
	if(tree_head_ == NULL)
		return ;
	FrontTree * cur = tree_head_;
	while(cur != NULL)
	{
		if(cur->front_tree_ == NULL)//termination node,delete
		{
			FrontTree *tmp = NULL;
			if(cur->next_ == NULL)//no next
			{
				tmp = cur ;
				cur = tmp->prev_tree_;
				delete tmp;
				--node_num_;
				if(cur != NULL)
					cur->front_tree_ = NULL;
				continue;
			}
			else
			{
				tmp = cur ;
				cur = tmp->next_;
				delete tmp;
				--node_num_;
				continue;
			}
			
		}
		cur = cur->front_tree_;
	}
	tree_head_ = NULL;
	return ;
}

float FrontTreeClass::CalcNgramRnn(Token *tok, int word_id,int prev_word_id)
{
	if(word_id == start_wordid_)
	{
		lm_->GetNgramScore((unsigned)word_id,tok->ngram_state_);
		return 0;
	}
	float lmscore = lm_->GetNgramScore((unsigned)word_id,tok->ngram_state_);
	tok->ngram_score_ += lmscore;
	lmscore = powf(10,lmscore);
	int prevword = ngramword_map_rnnword_[prev_word_id];
	int curword = ngramword_map_rnnword_[word_id];

	float rnnscore = 0;
	/*
	if(tok->rnn_class_out_.flag == 0)
	{
		rnnscore = rnn_->forword(prevword,curword,
				tok->rnn_hidden_out_.rnn_prev_hidden_out,
				tok->rnn_hidden_out_.rnn_prev_hidden_out_dim);
	}
	else*/
	{
		rnnscore = rnn_->forword(prevword,curword,
				tok->rnn_hidden_out_.rnn_prev_hidden_out,
				tok->rnn_hidden_out_.rnn_prev_hidden_out_dim,
				tok->rnn_class_out_.rnn_cur_class_out,
				tok->rnn_class_out_.class_dim, tok->rnn_class_out_.flag);
	}
	tok->rnn_class_out_.flag = 1;
	tok->rnn_score_ += log10f(rnnscore);
	tok->total_score_ += log10f(lmrate_ * lmscore + (1-lmrate_) * rnnscore) + penalty_/lmweight_;
#ifdef DEBUG
	++calc_num_ ;
	fprintf(stdout,"%5d %6d %6d %f %f \n",calc_num_,prev_word_id,word_id,lmscore,rnnscore);
#endif
	return tok->total_score_ * lmweight_;
}

void FrontTreeClass::Calc(Token *prev,FrontTree *curnode, int flag)
{
//	prev->flag_ = 1;
//	int flag = 0;//whether current node have been marked. 0 no,1 yes
	int word_id=curnode->word_id_;
	int prev_word_id=curnode->prev_tree_->word_id_;
	Token tok_cur(prev,flag);
	//calc score rnn and ngram
	//start
	float score = CalcNgramRnn(&tok_cur,word_id,prev_word_id);
	if(FLAGS == 1)
	{
		//prev hidden data should updata.because front node have been calculate,
		//so hidden and class out used for next point node
		memcpy(prev->rnn_hidden_out_.rnn_prev_hidden_out,
				tok_cur.rnn_hidden_out_.rnn_prev_hidden_out,
				sizeof(float)*tok_cur.rnn_hidden_out_.rnn_prev_hidden_out_dim);
		memcpy(prev->rnn_class_out_.rnn_cur_class_out,
				tok_cur.rnn_class_out_.rnn_cur_class_out,
				sizeof(float)*tok_cur.rnn_class_out_.class_dim);
	}
	//end
	//because some sentence is sort,so some middle node is end,FrontTree point
	//have add leaf,so must search leaf
	//at here ,</s> is end flag ,so it neet not search leaf
/*	if(leaf.find(curnode) != leaf.end())
	{
		leaf[curnode] += score;
		if(maxscore < leaf[curnode])
			maxscore = leaf[curnode];
		maxleafscore = curnode;
	}
*/	
#define LN_10 2.302585
	FrontTree *tree_cur = curnode;
	if(tree_cur->front_tree_ == NULL)
	{
		leaf[tree_cur].amscore_ += score * LN_10; 
#ifdef DEBUG
		nn ++;
		printf("%f %d %p %d\n",leaf[tree_cur].amscore_,leaf[tree_cur].best_num_,tree_cur,nn);
#endif
		if(maxscore_ < leaf[tree_cur].amscore_)
		{
			maxscore_ = leaf[tree_cur].amscore_;
			maxleafscore_ = tree_cur;
			max_socre_num_ = leaf[tree_cur].best_num_;
#ifdef DEBUG
			printf("%f %d %p\n",maxscore_,max_socre_num_,maxleafscore_);
			fflush(stdout);
#endif
		}
		/*
		 * here is no use.
		if(tree_cur->next_ != NULL)
		{
			printf("it's a bug.\n");
			Calc(prev, tree_cur->next_, FLAGS);//there should not run,if run ,it's bug.
		}
		else
		{
			return ;
		}*/
	}
	else
	{
		Calc(&tok_cur,curnode->front_tree_, 0);//front direction
	}
	if(tree_cur->next_ != NULL)//down direction
	{
#ifdef DEBUG
		fprintf(stderr,"sentence number %d\n",++sentence_num_);
#endif
		Calc(prev, tree_cur->next_, FLAGS);
	}
	/* //it's wrong.
	if(curnode->prev_tree_->next_ != NULL)
	{
#ifdef DEBUG
		fprintf(stderr,"sentence number %d\n",++sentence_num_);
#endif
		Calc(prev->prev_,curnode->prev_tree_->next_);
	}*/
	return ;
}
//lmrate: rnn and ngram rate
void FrontTreeClass::CalcScore(float lmrate,float lmweight,float penalty)
{
	lmrate_ = lmrate;
	lmweight_ = lmweight;
	penalty_ = penalty;
	Token tok_cur(rnn_->GetHiddenDim(),rnn_->GetClassDim());
	tok_cur.InitHidden(0.1);
	FrontTree *curtree = tree_head_;
	int word_id = tree_head_->word_id_;
	lm_->GetNgramScore((unsigned)word_id,tok_cur.ngram_state_);
	Calc(&tok_cur,curtree->front_tree_,0);
#ifdef DEBUG
	printf("leaf_node output\n");
	for(int i=0;i<leaf_node.size();++i)
		printf("%d %p %f %d\n",i,leaf_node[i],
				leaf[leaf_node[i]].amscore_,leaf[leaf_node[i]].best_num_);
#endif
	return ;
}
