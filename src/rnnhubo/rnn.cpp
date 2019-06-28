#include <string.h>
#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include "rnn.h"

void Rnn::LoadRNNLM(string modelname)
{
	LoadTextRNNLM(modelname);
}

void Rnn::LoadTextRNNLM (string modelname)
{
    int i, a, b;
    float v;
    char word[1024];
    FILE *fptr = NULL;
    // read model file
    fptr = fopen (modelname.c_str(), "r");
    if (fptr == NULL)
    {
        printf ("ERROR: Failed to read RNNLM model file(%s)\n", modelname.c_str());
        exit (0);
    }
    fscanf (fptr, "cuedrnnlm v%f\n", &v);
    if (v - 0.1 > 0.1 || v - 0.1 < -0.1)
    {
        printf ("Error: the version of rnnlm model(v%.1f) is not consistent with binary supported(v%.1f)\n", v, 0.1);
        exit (0);
    }
    fscanf (fptr, "train file: %s\n", word);     trainfile = word;
    fscanf (fptr, "valid file: %s\n", word);     validfile = word;
    fscanf (fptr, "number of iteration: %d\n", &iter);
    fscanf (fptr, "#train words: %d\n", &trainwordcnt);
    fscanf (fptr, "#valid words: %d\n", &validwordcnt);
    fscanf (fptr, "#layer: %d\n", &num_layer);
    layersizes.resize(num_layer+1);
    for (i=0; i<layersizes.size(); i++)
    {
        fscanf (fptr, "layer %d size: %d\n", &b, &a);
        assert(b==i);
        layersizes[i] = a;
    }
    fscanf (fptr, "feature dimension: %d\n", &dim_fea);
    fscanf (fptr, "class layer dimension: %d\n", &nclass);
    allocMem();

    fscanf (fptr, "independent mode: %d\n", &independent);
    fscanf (fptr, "train crit mode: %d\n",  &traincritmode);
    for (i=0; i<num_layer; i++)
    {
        fscanf (fptr, "layer %d -> %d\n", &a, &b);
        assert (a==i);
        assert (b==(i+1));
        for (a=0; a<layersizes[i]; a++)
        {
            for (b=0; b<layersizes[i+1]; b++)
            {
                fscanf (fptr, "%f", &v);
                if(0 != layers[i]->SetData(a, b, v))
				{
					fprintf(stderr,"matrix SetData error!\n");
					return ;
				}
            }
            fscanf (fptr, "\n");
        }
    }
	
    fscanf (fptr, "recurrent layer 1 -> 1\n");
    for (a=0; a<layersizes[1]; a++)
    {
        for (b=0; b<layersizes[1]; b++)
        {
            fscanf (fptr, "%f", &v);
            if(0 != hidtohid_weight[0]->SetData(a, b, v))
			{
				fprintf(stderr,"matrix SetData error!\n");
				return ;
			}
        }
        fscanf (fptr, "\n");
    }
	/*
    if (dim_fea > 0)
    {
        fscanf (fptr, "feature layer weight\n");
        for (a=0; a<dim_fea; a++)
        {
            for (b=0; b<layersizes[1]; b++)
            {
                fscanf (fptr, "%f", &v);
                layer0_fea->assignhostvalue(a, b, v);
            }
            fscanf (fptr, "\n");
        }
    }*/
    if (nclass > 0)
    {
        fscanf (fptr, "class layer weight\n");
        for (a=0; a<layersizes[num_layer-1]; a++)
        {
            for (b=0; b<nclass; b++)
            {
                fscanf (fptr, "%f", &v);
                class_layer_weight->SetData(a, b, v);
            }
            fscanf (fptr, "\n");
        }
    }
    fscanf (fptr, "hidden layer ac\n");
    for (a=0; a<layersizes[1]; a++)
    {
        fscanf (fptr, "%f", &v);
        for (b=0; b<minibatch; b++) 
			neu0_ac_hist->SetData(a, b, v);
    }
    fscanf (fptr, "\n");
    fscanf (fptr, "%d", &a);
    if (a != 9999999)
    {
        printf ("ERROR: failed to read the check number(%d) when reading model\n",9999999);// CHECKNUM);
        exit (0);
    }
 //   if (debug > 1)
    {
        printf ("Successfully loaded model: %s\n", modelname.c_str());
    }
    fclose (fptr);
}

void Rnn::allocMem()
{
	hidtohid_weight.resize(1);
	hidtohid_weight[0] = new Matrix(layersizes[1], layersizes[1]);
	layers.resize(num_layer);
	for(int i=0;i<num_layer;++i)
	{
		layers[i] = new Matrix(layersizes[i],layersizes[i+1]);
	}
	if(nclass > 0)
	{
		class_layer_weight = new Matrix(layersizes[num_layer-1], nclass);
	}
	neu0_ac_hist = new Matrix(layersizes[1],1);
}
//only calculate word output layer.
float Rnn::forword(int curword,float *hidden,float hiddim,
		float *class_out,int class_dim,float *output, int outdim)
{
	memset(output,0x00,sizeof(float)*outdim);
	int nrow = layersizes[num_layer-1];
	//int ncol = layersizes[num_layer];
	int clsid = word2class[curword];
	int swordid = classinfo[clsid*3];
	int nword = classinfo[clsid*3+2];
	float *wgt = layers[num_layer-1]->GetDataP();
	matrixXvector(hidden,wgt+swordid*nrow,output+swordid, nrow, nword);
	SoftMax(output+swordid,nword);
	if(nclass > 0)
	{
		int clsid = word2class[curword];
		return output[curword] * class_out[clsid];
	}
	else
	{
		return output[curword];
	}
}
//iodim: iput and oput dim ,it's word list length.
//iput,oput,hidden_out:external application for memory.

float Rnn::forword (int prevword, int curword,float *iput,float *oput,int iodim,
	  float *hid_in,int hid_dim,float *class_out,int class_dim)
{
	int a, b;
   	//int c;
	int nrow, ncol;
	//float v, norm, maxv;
	nrow = layersizes[1];
	ncol = layersizes[1];
	float *wgt;
	//iodim == layersizes[0]
	//hidden_out == layersizes[1]
	//class_dim == nclass
	memset(iput,0x00,sizeof(float)*iodim);
	memset(oput,0x00,sizeof(float)*iodim);
	memset(class_out,0x00,sizeof(float)*class_dim);
	// neu0 -> neu1
	for (a=0; a<layers.size(); a++)
	{
		if (a==0)//calculate feature * weight,first layer
		{
			memcpy(iput,hid_in ,hid_dim*sizeof(float));
			wgt   = hidtohid_weight[0]->GetDataP();
//			oput = neu_ac[1]->gethostdataptr();
			nrow  = layersizes[1];
			ncol  = layersizes[1];
			memset(oput, 0, sizeof(float)*ncol);
			for(b=0;b<ncol;++b)
			{
				oput[b] = layers[0]->GetData(prevword,b);
			}
		}
		else//last layer
		{
			if(a==1)
				memcpy(hid_in,oput,hid_dim*sizeof(float));
			float *tmp=NULL;
			tmp = iput;
			iput = oput;
			oput = tmp;
			nrow  = layersizes[a];
			ncol  = layersizes[a+1];
			memset(oput, 0, sizeof(float)*ncol);
			wgt = layers[a]->GetDataP();
		}
		if (a+1==num_layer)
		{
			if(lognormconst < 0)
			{
				if (nclass > 0)
				{
					int ncol_cls = nclass;
					float *cls_w = class_layer_weight->GetDataP();
					matrixXvector(iput,cls_w,
							class_out,nrow, ncol_cls);
					//for(int i=0;i<ncol_cls;++i)
					SoftMax(class_out,ncol_cls);
					int clsid = word2class[curword];
					int swordid = classinfo[clsid*3];
					//int ewordid = classinfo[clsid*3+1];
					int nword   = classinfo[clsid*3+2];
					matrixXvector(iput, wgt+swordid*nrow, oput+swordid, nrow, nword);
					SoftMax(oput+swordid,nword);

				}
				else
				{
					matrixXvector(iput,wgt,oput,nrow, ncol);
					SoftMax(oput,ncol);
				}
			}
			else
			{
				float v = 0;
				for (int i=0; i<nrow; i++)
				{
					v += iput[i]*layers[a]->GetData(i,curword);
				}
				oput[curword] = exp(v-lognormconst);
			}
		}
		else
		{
			matrixXvector (iput, wgt, oput, nrow, ncol);
			for(int i=0;i<ncol;++i)
				oput[i]=SigMoid(oput[i]);
		}
	}

	if(nclass > 0)
	{
		int clsid = word2class[curword];
		return oput[curword] * class_out[clsid];
	}
	else
	{
		return oput[curword];
	}
}
float Rnn::SigMoid(float A)
{
	return 1/(1+exp(-A));
}
void Rnn::SoftMax(float *A,int dim)
{
	int a, maxi;
	float v, norm, maxv = 1e-8;
	maxv = 1e-10;
	for (a=0; a<dim; a++)
	{
		v = A[a];
		if(v > maxv)
		{
			maxv = v;
			maxi = a;
		}
	}
	norm = 0;
	for (a=0; a<dim; a++)
	{
		v = A[a] - maxv;
		A[a] = exp(v);
		norm += A[a];
	}
	for (a=0; a<dim; a++)
	{
		v = A[a] / norm;
		A[a] = v;
	}
}

void Rnn::matrixXvector(float *src, float *wgt, float *dst, int nr, int nc)
{
#ifndef CBLAS
	int i,j;
	for (i=0; i<nc; i++)
	{
		for (j=0; j<nr; j++)
		{
			dst[i] += wgt[j+i*nr]*src[j];
		}
	}
#else
	cblas_sgemm(CblasRowMajor,CblasNoTrans,CblasTrans,
			1,nc,nr,1.0,(float const *)src,nr,(float const *)wgt,nr,1.0,dst,nc);
#endif
	return ;
}

void Rnn::ReadWordlist(string inputlist, string outputlist)
{
	int i;
   	//int a, b;
	//float v;
	char word[1024];
	FILE *finlst, *foutlst;
	finlst = fopen (inputlist.c_str(), "r");
	foutlst = fopen (outputlist.c_str(), "r");
	if (finlst == NULL || foutlst == NULL)
	{
		printf ("ERROR: Failed to open input (%s) or output list file(%s)\n", inputlist.c_str(), outputlist.c_str());
		exit (0);
	}
	inputmap.insert(make_pair(string("<s>"), 0));
	outputmap.insert(make_pair(string("</s>"), 0));
	inputvec.clear();
	outputvec.clear();
	inputvec.push_back("<s>");
	outputvec.push_back("</s>");
	int index = 1;
	while (!feof(finlst))
	{
		if(fscanf (finlst, "%d%s", &i, word) == 2)
		{
			if (inputmap.find(word) == inputmap.end())
			{
//				printf("%d %s\n",i,word);
				inputmap[word] = index;
				inputvec.push_back(word);
				index ++;
			}
		}
	}
	if (inputmap.find("<OOS>") == inputmap.end())
	{
		inputmap.insert(make_pair(string("<OOS>"), index));
		inputvec.push_back("<OOS>");
	}
	else
	{
		assert (inputmap["<OOS>"] == inputvec.size()-1);
	}

	index = 1;
	// allocate memory for class information
	if (nclass > 0)
	{
		word2class = new int[layersizes[num_layer]];
		classinfo = new int[nclass*3];
		classinfo[0] = 0;
	}
	int clsid, prevclsid = 0;
	while (!feof(foutlst))
	{
		if (nclass > 0)
		{
			if (fscanf(foutlst, "%d%s%d", &i, word, &clsid) == 3)
			{
//				printf("%d %s\n",i,word);
				if (outputmap.find(word) == outputmap.end())
				{
					outputmap[word] = index;
					outputvec.push_back(word);
					index ++;
				}
				int idx = outputmap[word];
				word2class[idx] = clsid;
				if (clsid != prevclsid)
				{
					classinfo[prevclsid*3+1] = idx-1;
					classinfo[prevclsid*3+2] = idx-classinfo[prevclsid*3];
					classinfo[3*clsid]=idx;
				}
				prevclsid = clsid;
			}
		}
		else
		{
			if (fscanf(foutlst, "%d%s", &i, word) == 2)
			{
				if (outputmap.find(word) == outputmap.end())
				{
					 outputmap[word] = index;
					 outputvec.push_back(word);
					 index ++;
				}
			}
		}
	}
	if (nclass > 0)
	{
		classinfo[prevclsid*3+1] = layersizes[num_layer]-1;
		classinfo[prevclsid*3+2] = layersizes[num_layer]-classinfo[prevclsid*3];
	}
	if (outputmap.find("<OOS>") == outputmap.end())
	{
		outputmap.insert(make_pair(string("<OOS>"), index));
		outputvec.push_back("<OOS>");
	}
	else
	{
		 assert (outputmap["<OOS>"] == outputvec.size()-1);
	}
	fprintf(stderr,"inputvec %ld layersizes[0] %d\n", inputvec.size(), layersizes[0]);
	assert (inputvec.size() == layersizes[0]);
	assert (outputvec.size() == layersizes[num_layer]);
	inStartindex = 0;
	outEndindex  = 0;
	inOOSindex   = inputvec.size() - 1;
	outOOSindex  = outputvec.size() - 1;
	assert (outOOSindex == outputmap["<OOS>"]);
	assert (inOOSindex == inputmap["<OOS>"]);
	fclose (finlst);
	fclose (foutlst);
}

