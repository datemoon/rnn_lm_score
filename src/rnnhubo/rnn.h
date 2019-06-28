#ifndef __LM_RNN__
#define __LM_RNN__
#include <string.h>
#include <string>
#include <vector>
#include <map>
#include <math.h>
#include "matrix.h"
#ifdef CBLAS
extern "C"
{
#include "cblas.h"
}
#endif
using namespace std;
class Rnn
{
public:
	Rnn(){minibatch=1;lognormconst=-1.0;}
	~Rnn()
	{
		delete hidtohid_weight[0];
		for(int i=0;i<num_layer;++i)
			delete layers[i];
		delete class_layer_weight;
		delete neu0_ac_hist;
		delete[] word2class;
		delete[] classinfo;
	}
	void LoadRNNLM(string modelname);
	//only calculate output layer.
	float forword(int curword,float *hidden,float hiddim,
			float *class_in,int class_dim,float *output, int outdim);
	float forword (int prevword, int curword,float *iput,float *oput,int iodim,
			float *hid_in,int hid_dim,float *class_out,int class_dim);
	void ReadWordlist(string inputlist, string outputlist);
	int GetWordId(string word)
	{
		if(outputmap.find(word) ==  outputmap.end())
			return outOOSindex;
		else
			return outputmap[word];
	}
	int GetStartIndex(){return inStartindex;}
	int GetEndIndex(){return outEndindex;}
	int GetNclass(){return nclass;}
	int GetMaxLayerNode()
	{
		int max=0;
		for(int i=0;i<layersizes.size();++i)
			max = max > layersizes[i]? max:layersizes[i];
		return max;
	}
	int GetHidDim(){return layersizes[1];}
private:
	void LoadTextRNNLM (string modelname);
	void allocMem();
	void matrixXvector(float *src, float *wgt, float *dst, int nr, int nc);
	float SigMoid(float A);
	void SoftMax(float *A,int dim);

private:
	string inmodelfile, outmodelfile, trainfile, validfile,
		testfile, inputwlist, outputwlist, nglmstfile,
		sampletextfile, uglmfile, feafile;
	map<string, int> inputmap,outputmap;
	vector<string>  inputvec, outputvec, ooswordsvec;
	int *classinfo;
	int *word2class;
	int inStartindex,outEndindex,inOOSindex,outOOSindex;
private:
	//rnn parameter
	int iter,trainwordcnt,validwordcnt,independent,traincritmode,minibatch;
	int dim_fea;
	float lognormconst;//-1.0
	int nclass;
	int num_layer;//layer number
	vector<int> layersizes;//every layers node number
	vector<Matrix *> hidtohid_weight;
	vector<Matrix *> layers;
	Matrix *class_layer_weight;
	Matrix *neu0_ac_hist;
};

class RnnCalc
{
public:
	RnnCalc(Rnn *rnn)
	{
		rnn_ = rnn;
		ioputdim_ = rnn_->GetMaxLayerNode();
		classdim_ = rnn_->GetNclass();
		hiddim_ = rnn_->GetHidDim();
		rnninput_ = new float[ioputdim_];
		rnnoutput_ = new float[ioputdim_];
		classoutput_ = new float[classdim_];
		hidin_ = new float[hiddim_];
	}
	~RnnCalc()
	{
		delete[] rnninput_;delete[] rnnoutput_;
		delete[] classoutput_;delete[] hidin_;
	}
	float forword(int prevword,int curword,float *hidin,int hiddim , float *class_in ,int class_dim ,int flag = 0)
	{
		if(hiddim != hiddim_ || class_dim != classdim_)
		{
			fprintf(stderr,"hiddim(%d) or class_dim(%d) is different",
					hiddim,class_dim);
			return -1;
		}
		float rnnscore = 0;
		memcpy(hidin_,hidin,sizeof(float)*hiddim_);
		if(flag == 0)
		{
			rnnscore = rnn_->forword(prevword,curword,rnninput_,rnnoutput_,ioputdim_,
					hidin_,hiddim_,classoutput_,classdim_);
			memcpy(class_in,classoutput_,sizeof(float)*class_dim);
		}
		else
		{
			memcpy(classoutput_,class_in,sizeof(float)*class_dim);
			rnnscore = rnn_->forword(curword,hidin_,hiddim_,classoutput_,class_dim,rnnoutput_,ioputdim_);

		}
		memcpy(hidin,hidin_,sizeof(float)*hiddim_);
		return rnnscore;
	}
	int GetHiddenDim(){ return hiddim_;}
	int GetClassDim(){ return classdim_;}
private:
	Rnn *rnn_;
	float *rnninput_;
	float *rnnoutput_;
	float *classoutput_;
	float *hidin_;
	int ioputdim_;
	int classdim_;
	int hiddim_;
};

#endif
