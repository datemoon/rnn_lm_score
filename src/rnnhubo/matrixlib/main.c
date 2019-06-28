#include <stdio.h>
#include <sys/time.h>
#include "cblas.h"

#define ROW 50
#define COL 600
int main(int argc,char *argv[])
{
//	float A[ROW][COL] = {{1,2,3,4},{3,4,5,6},{5,6,7,8}};
//	float B[ROW][COL] = {{1,1,1,1},{2,2,2,2},{3,3,3,3}};

//	float A[ROW][COL] = {{1.0,2.0,3.0},{4.0,5.0,6.0},{7.0,8.0,9.0}};
//	float B[ROW][ROW] = {{1.0,1.0,1.0},{1.0,1.0,1.0},{1.0,1.0,1.0}};
//	float C[ROW][ROW] = {{1.0,1.0,1.0},{1.0,1.0,1.0},{1.0,1.0,1.0}};
	float A[ROW][COL];
	float B[COL][COL];
	float C[ROW][COL];
	int i=0,j=0;
	int k = 0;
	for(i=0;i<ROW;++i)
	{
		for(j=0;j<COL;++j)
		{
			A[i][j]=1,C[i][j]=1;
		}
	}
	for(i=0;i<COL;++i)
	{
		for(j=0;j<COL;++j)
			B[i][j]=1;
	}
	struct timeval start,end;
	gettimeofday(&start,NULL);
	while(k++<10)
	cblas_sgemm(CblasRowMajor,CblasNoTrans,CblasTrans,ROW,COL,COL,1.0,A[0],COL,B[0],COL,1.0,C[0],COL);
	gettimeofday(&end,NULL);
	printf("time is %f\n",end.tv_sec-start.tv_sec+
			(end.tv_usec - start.tv_usec)*1.0/1000000);
	gettimeofday(&start,NULL);
	k=0;
	while(k++<10)
	for(i=0;i<ROW;++i)
	{
		cblas_sgemm(CblasRowMajor,CblasNoTrans,CblasTrans,1,COL,COL,1.0,A[0],COL,B[0],COL,1.0,C[i],COL);
	}
	gettimeofday(&end,NULL);
	printf("time is %f\n",end.tv_sec-start.tv_sec+
			(end.tv_usec - start.tv_usec)*1.0/1000000);
#ifdef DEBUG
	for(i=0;i<ROW;++i)
	{
		for(j=0;j<COL;++j)
		{
			printf("%6.1f ",C[i][j]);
		}
		printf("\n");
	}
#endif
	return 0;
}
