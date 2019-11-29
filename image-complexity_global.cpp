//histGray is used to calculate the entropy of a grayscale image using OpenCV calcHis

#include "stdafx.h"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include <iostream>
#include <stdio.h>
#include <vector>
#include <io.h>
#include <fstream>
using namespace std;
using namespace cv;

#define pi 3.14159265358979323846
double MSE_PSNR(int rows, int cols, Mat image, Mat shift_image);
double L(int rows, int cols, Mat image, Mat shift_image);
double C(int rows, int cols, Mat image, Mat shift_image);
double S(int rows, int cols, Mat image, Mat shift_image);
int i, j;
double psnr;
//LUMINANCE����
double mean_as;//sigma image & shift_image
double mean_a;
double mean_s;
double luminance;
//CONTRAST����
double contrast_a;
double contrast_s;
double contrast;

/**
   PSNR+SSIM
   SSIM ԽСԽ����
   PSNR 
**/
double MSE_PSNR(int rows, int cols, Mat image, Mat shift_image)
{//PSNR ��ֵԽ��ͼ��ʧ��ԽС��ͼ������Խ�ߣ������Ϊ���Ӷ�Խ��
	double difference = 0;
	double sigma = 0;
	double mse;
	for (i = 0; i<rows; i++)
	{
		for (j = 0; j<cols; j++)
		{
			(double)difference = (double)image.at<unsigned char>(i, j) - (double)shift_image.at<unsigned char>(i, j);
			sigma = sigma + (double)difference*(double)difference;
		}
	}
	mse = sigma / (rows*cols);
	psnr = 10 * log10(255 * 255 / mse);//PSNR
									   //printf("PSNR=%f\n", psnr);
	psnr = 1/(psnr / 1000)-18;		//���Ӷȹ�һ��					  
	return psnr;
}

double L(int rows, int cols, Mat image, Mat shift_image)
{//����
	double sum_a = 0;
	double sum_s = 0;
	double C1 = 0.1;
	for (i = 0; i<rows; i++)
	{
		for (j = 0; j<cols; j++)
		{
			sum_a = sum_a + (double)image.at<unsigned char>(i, j);
			sum_s = sum_s + (double)shift_image.at<unsigned char>(i, j);
		}
	}
	mean_a = sum_a / (rows*cols);
	mean_s = sum_s / (rows*cols);;
	luminance = (2 * mean_a*mean_s + C1) / (mean_a*mean_a + mean_s*mean_s + C1);
	//printf("\nLUMINANCE=%f\n", luminance);
	return luminance*3;
}

double C(int rows, int cols, Mat image, Mat shift_image)
{//�Աȶ�
	double variance_a = 0;//image׃����
	double variance_s = 0;//shift_image׃����
	int C2 = 0.1;
	for (i = 0; i<rows; i++)
	{
		for (j = 0; j<cols; j++)
		{
			variance_a = variance_a + ((double)image.at<unsigned char>(i, j) - mean_a)
				*((double)image.at<unsigned char>(i, j) - mean_a);
			variance_s = variance_s + ((double)shift_image.at<unsigned char>(i, j) - mean_s)
				*((double)shift_image.at<unsigned char>(i, j) - mean_s);
		}
	}
	contrast_a = sqrt(variance_a / (rows*cols));
	contrast_s = sqrt(variance_s / (rows*cols));
	contrast = ((2 * contrast_a*contrast_s) + C2) / (contrast_a*contrast_a + contrast_s*contrast_s + C2);
	//printf("CONTRAST=%f\n", contrast);
	return contrast*10;
}

double S(int rows, int cols, Mat image, Mat shift_image)
{//�ṹ
	double variance_as = 0;
	double contrast_as;
	double structure;
	double C3 = 0.1;
	for (i = 0; i<rows; i++)
	{
		for (j = 0; j<cols; j++)
		{
			variance_as = variance_as + (image.at<unsigned char>(i, j) - mean_a)
				*(shift_image.at<unsigned char>(i, j) - mean_s);
		}
	}
	contrast_as = variance_as / (rows*cols - 1);
	structure = (contrast_as + C3) / (contrast_a*contrast_s + C3);
	//printf("STRUCTURE=%f\n", structure);
	return 2*structure;

}

double PSNR_SSIM(Mat image){
	//SSIMȡֵ��Χ[0,1]��ֵԽ�󣬱�ʾͼ��ʧ��ԽС.����Խ�ߣ�����Ϊ���Ӷȵ�
    int cols = image.cols;//width
	int rows = image.rows;//height
	double l;
	double c;
	double s;
	double ssim;
	Mat shift_image = Mat(rows, cols, CV_8UC1);
	for (j = 0; j<cols - 1; j++)//����
	{
		for (i = 0; i<rows; i++)
		{
			if (j == 0)
			{
				shift_image.at<unsigned char>(i, j + 1) = image.at<unsigned char>(i, j);
				shift_image.at<unsigned char>(i, j) = image.at<unsigned char>(i, j);
			}
			else shift_image.at<unsigned char>(i, j + 1) = image.at<unsigned char>(i, j);
		}
	}
	MSE_PSNR(rows, cols, image, shift_image);
	l = L(rows, cols, image, shift_image);
	c = C(rows, cols, image, shift_image);
	s = S(rows, cols, image, shift_image);
	ssim = l*c*s;//ssim
	ssim = 1000/ssim-10;//���Ӷ���ֵ��һ��
	cout<<"ssim: "<<ssim<<endl;
	cout<<"psnr: "<<psnr<<endl;

	return ssim;
}


/**
   ��������
   ����Խ��Խ����
**/
double EstimateNoise(Mat img)
{
	Mat kern = (Mat_<char>(3, 3) << 1, -2, 1,
		-2, 4, -2,
		1, -2, 1);
	Mat CImage;
	filter2D(img, CImage, img.depth(), kern);
	//abs(dstImage);
	//imshow("test",dstImage);
	Mat temp;
	int nr = CImage.rows;
	int nc = CImage.cols;
	double Sigma;
	int sum=0;
	for (int i = 0; i < nr; i++)
	{
		for (int j = 0; j < nc; j++)
		{
			sum+=CImage.at<uchar>(i, j);
		}
	}
	//cout << sum << endl;
	Sigma= sum*sqrt(0.5*pi) / (6 * (nr - 2)*(nc - 2));
	cout << "Noise: "<<Sigma << endl;
	Sigma = Sigma * 50;
	return Sigma;
}

/**
   Engropy ��Ϣ��
   ��Ϣ��Խ��Խ����
**/
double Entropy(Mat img)
{
   // ������ľ���Ϊͼ��
   double temp[256];

   double num[256];
   int sum=0;
   double var=0;
   double aver=0;

   // ����
   for(int i=0;i<256;i++)
   {
    temp[i] = 0.0;
    num[i]=0;
   }
 
   // ����ÿ�����ص��ۻ�ֵ
   for(int m=0;m<img.rows;m++)
   {// ��Ч�������еķ�ʽ
    const uchar* t = img.ptr<uchar>(m);
    for(int n=0;n<img.cols;n++)
    {
     int i = t[n];
     temp[i] = temp[i]+1;
     sum+=img.at<uchar>(m,n);
    }
   }
 
   aver=sum/(img.rows*img.cols);
   for(int i=0;i<img.rows;i++)
	  for(int j=0;j<img.cols;j++){
	     var+=(img.at<uchar>(i,j)-aver)*(img.at<uchar>(i,j)-aver);
	  }
   var=var/(img.rows*img.cols);

   // ����ÿ�����صĸ���
   for(int i=0;i<256;i++)
   {
    temp[i] = temp[i]/(img.rows*img.cols);
   }
   double result1 = 0;
   double result2=0;    //��Ȩ��Ϣ��
   double result3=0;    //�����Ȩ��Ϣ��
   // ���ݶ������ͼ����
   for(int i =0;i<256;i++)
   {
    if(temp[i]==0.0)
     {
	   result1 = result1;
	   result2=result2;
	   result3=result3;
     }
    else
     {
	   result1 = result1-temp[i]*(log(temp[i])/log(2.0));   //��Ϣ��
	   result2=result2-i*temp[i]*(log(temp[i])/log(2.0));   //��Ȩ��Ϣ��
	   result3=result3-var*temp[i]*(log(temp[i])/log(2.0));  //�����Ȩ��Ϣ��
     }
 }

   cout<<"��Ϣ��: "<<result1<<endl;
   cout<<"��Ȩ��Ϣ��: "<<result2<<endl;
   cout<<"������Ϣ��: "<<result3<<endl;
   return result1;//*img.rows*img.cols; 
}

void getFiles(string path,vector<string>&files){
  //�ļ����  
    long   hFile   =   0;  
    //�ļ���Ϣ  
    struct _finddata_t fileinfo;  
    string p;  
    if((hFile = _findfirst(p.assign(path).append("\\*").c_str(),&fileinfo)) !=  -1)  
    {  
        do  
        {  
            //�����Ŀ¼,����֮  
            //�������,�����б�  
            if((fileinfo.attrib &  _A_SUBDIR))  
            {  
                if(strcmp(fileinfo.name,".") != 0  &&  strcmp(fileinfo.name,"..") != 0)  
                    getFiles( p.assign(path).append("\\").append(fileinfo.name), files );  
            }  
            else  
            {  
                files.push_back(p.assign(path).append("\\").append(fileinfo.name) );  
            }  
        }while(_findnext(hFile, &fileinfo)  == 0);  
        _findclose(hFile);  
    }  
}

int main( int, char** argv )
{

  vector<string> files;
  string path="image";
  double entro,ssim,noise;
  double a=0.2,b=0.2,c=0.3;//c=
  double rlt;
  /*FILE *temp;
  temp = fopen("rlt.txt", "w");*/

  ofstream f("rlt.txt");
  if(!f) return 0;
  getFiles(path,files);
  if(files.size()!=0){
	  for(vector<string>::iterator it=files.begin();it!=files.end();it++){
	     string file_path=*it;
		  /// Load image
		 Mat src;
         //src = imread("D:\\b.jpg");   
		 src = imread(file_path); 
         Mat gray;
         cvtColor(src, gray, 7);//ת���ɻҶ�ͼ  
         entro=Entropy(src);  //entropy ��Ϣ��
         ssim=PSNR_SSIM(src);  //psnr �� ssim
         noise=EstimateNoise(src);

         rlt=a*entro+b*ssim+c*psnr+(1-a-b-c)*noise-2;   //�������
		 //fprintf(temp, "entropy=%f,ssim=%f,psnr=%f,noise=%f,result=%rlt\n", entro,ssim,psnr,noise,rlt);
		 f<<"FileName="<<file_path<<",  entropy="<<entro<<"	 "<<",ssim="<<ssim << "	 " <<",psnr="<<psnr << "  " <<",noise="<<noise << " " <<",result="<<rlt<<endl;
		 //�����ļ�
         cout<<rlt<<endl;
         src.release();
	  }
  }
  
  waitKey(0);
  return 0;

}