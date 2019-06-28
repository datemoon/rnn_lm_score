#ifndef __MATRIX_LM_H_
#define __MATRIX_LM_H_

typedef float Float;

class Matrix
{
public:
	Matrix(int row,int col):rows_(row),cols_(col)
	{
		size_ = row*col;
		data_ = new Float[size_];
	}
	~Matrix()
	{
		rows_=0;cols_=0;size_=0;delete[] data_;
	}
	int SetData(int row,int col,Float data)
	{
		if(row*col > size_)
			return -1;
		//data_[row*cols_+col] = data;
		data_[row+col*rows_] = data;
		return 0;
	}
	float GetData(int row,int col)
	{
		if(row*col > size_)
			return -1;
		return data_[row + col*rows_];
	}
	Float *GetDataP()
	{
		return data_;
	}
private:
	Float *data_;
	int rows_;
	int cols_;
	int size_;
};

#endif
