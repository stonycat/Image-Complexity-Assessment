//histGray is used to calculate the entropy of a grayscale image using OpenCV calcHis

#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/core/core.hpp"
#include <iostream>
#include <stdio.h>
#include <vector>
#include <io.h>
#include <cv.h>
#include <fstream>
#include <stdlib.h>
#include <string>
#include<math.h>

using namespace std;
using namespace cv;
double sqrt(double n);
#define pi 3.14159265358979323846
double MSE_PSNR(int rows, int cols, Mat image, Mat shift_image);
int i, j;
double psnr;
/*double L(int rows, int cols, Mat image, Mat shift_image);
double C(int rows, int cols, Mat image, Mat shift_image);
double S(int rows, int cols, Mat image, Mat shift_image);

//LUMINANCE����
double mean_as;//sigma image & shift_image
double mean_a;
double mean_s;
double luminance;
//CONTRAST����
double contrast_a;
double contrast_s;
double contrast;*/



/**PSNR+SSIM SSIM ԽСԽ���� PSNR**/
double MSE_PSNR(Mat image)
{//PSNR ��ֵԽ��ͼ��ʧ��ԽС��ͼ������Խ�ߣ������Ϊ���Ӷ�Խ��
	int cols = image.cols;//width
	int rows = image.rows;//height
	Mat shift_image = Mat(rows, cols, CV_8UC1);
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
	psnr = 1 / psnr+6;		//���Ӷȹ�һ��		
	
	return psnr;
}

/*double L(int rows, int cols, Mat image, Mat shift_image)
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
	return luminance;
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
	return contrast * 10;
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
	return 2 * structure;

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
	ssim = 100 / ssim - 1;//���Ӷ���ֵ��һ��
	cout << "ssim: " << ssim << endl;
	cout << "psnr: " << psnr << endl;

	return ssim;
}*/    //ssim ��ʱͣ��

/**�������� ����Խ��Խ����**/
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
	int sum = 0;
	for (int i = 0; i < nr; i++)
	{
		for (int j = 0; j < nc; j++)
		{
			sum += CImage.at<uchar>(i, j);
		}
	}
	//cout << sum << endl;
	Sigma = sum*sqrt(0.5*pi) / (6 * (nr - 2)*(nc - 2));
	
	Sigma = Sigma * 20;
	cout << "Noise: " << Sigma << endl;
	return Sigma;
}

/**Engropy ��Ϣ�� ��Ϣ��Խ��Խ����**/
double Entropy(Mat img)
{
	// ������ľ���Ϊͼ��
	double temp[256];

	double num[256];
	int sum = 0;
	double var = 0;
	double aver = 0;

	// ����
	for (int i = 0; i<256; i++)
	{
		temp[i] = 0.0;
		num[i] = 0;
	}

	// ����ÿ�����ص��ۻ�ֵ
	for (int m = 0; m<img.rows; m++)
	{// ��Ч�������еķ�ʽ
		const uchar* t = img.ptr<uchar>(m);
		for (int n = 0; n<img.cols; n++)
		{
			int i = t[n];
			temp[i] = temp[i] + 1;
			sum += img.at<uchar>(m, n);
		}
	}

	aver = sum / (img.rows*img.cols);
	for (int i = 0; i<img.rows; i++)
	for (int j = 0; j<img.cols; j++){
		var += (img.at<uchar>(i, j) - aver)*(img.at<uchar>(i, j) - aver);
	}
	var = var / (img.rows*img.cols);

	// ����ÿ�����صĸ���
	for (int i = 0; i<256; i++)
	{
		temp[i] = temp[i] / (img.rows*img.cols);
	}
	double result1 = 0;
	double result2 = 0;    //��Ȩ��Ϣ��
	double result3 = 0;    //�����Ȩ��Ϣ��
	// ���ݶ������ͼ����
	for (int i = 0; i<256; i++)
	{
		if (temp[i] == 0.0)
		{
			result1 = result1;
			result2 = result2;
			result3 = result3;
		}
		else
		{
			result1 = result1 - temp[i] * (log(temp[i]) / log(2.0));   //��Ϣ��
			result2 = result2 - i*temp[i] * (log(temp[i]) / log(2.0));   //��Ȩ��Ϣ��
			result3 = result3 - var*temp[i] * (log(temp[i]) / log(2.0));  //�����Ȩ��Ϣ��
		}
	}

	//cout << "��Ϣ��: " << result1 << endl;
	/*cout << "��Ȩ��Ϣ��: " << result2 << endl;
	cout << "������Ϣ��: " << result3 << endl;*/
	
	return result1;//*img.rows*img.cols; 
}

double Entropy_local(Mat img,Mat imgbg){
	cout << "local_Entropy: " << 10/(abs(Entropy(img) - Entropy(imgbg))) << endl;
	return abs(10 /(abs(Entropy(img) - Entropy(imgbg))))-5;
}

double RSS_local(Mat img, Mat imgbg)
{
	Mat tmp_m, tmp_sd;
	double ut = 0, ubg = 0, ot = 0;

	ut = mean(img)[0];
	ubg = mean(imgbg)[0];

	meanStdDev(img, tmp_m, tmp_sd);
	ut = tmp_m.at<double>(0, 0);
	ubg = tmp_m.at<double>(0, 0);
	ot = tmp_sd.at<double>(0, 0);
	double RSS_L = (sqrt(pow((ut - ubg), 2) + pow(ot, 2)))/10;
	cout <<"local_RSS: " << 1/RSS_L<< endl;
	return 1/RSS_L;

}

double Canny_local(Mat img, Mat imgbg)
{
	//Ŀ�����޶��ڽ�����ı�Ե����  ֵԽ��Խ����ȡĿ�꣬���Ӷ�Խ��
	Mat  edge;
	edge.create(img.size(), img.type());	// ������srcͬ���ͺʹ�С�ľ���
											// ����Canny����  
	Canny(img, edge, 150, 100, 3);
	double sum_p = 0.0;
	double M = double(imgbg.rows), N = double(imgbg.cols);
	Mat_<uchar>::iterator it = edge.begin<uchar>();
	Mat_<uchar>::iterator itend = edge.end<uchar>();
	for (; it != itend; it++)
	{
		if (*it == 255)
			sum_p++;
	}
	double canny_p = (sum_p / (M*N));
	//��ʾЧ��ͼ   
	imshow("Canny��Ե���", edge);

	canny_p = 1 / canny_p - 20;
	canny_p = 10 / (1 + exp(-canny_p + 1));

	cout << "canny_l: " << canny_p<< endl;
	return canny_p;//��Ե����
}

void getFiles(string path, vector<string>&files){
	//�ļ����  
	long   hFile = 0;
	//�ļ���Ϣ  
	struct _finddata_t fileinfo;
	string p;
	if ((hFile = _findfirst(p.assign(path).append("\\*").c_str(), &fileinfo)) != -1)
	{
		do
		{
			//�����Ŀ¼,����֮  
			//�������,�����б�  
			if ((fileinfo.attrib &  _A_SUBDIR))
			{
				if (strcmp(fileinfo.name, ".") != 0 && strcmp(fileinfo.name, "..") != 0)
					getFiles(p.assign(path).append("\\").append(fileinfo.name), files);
			}
			else
			{
				files.push_back(p.assign(path).append("\\").append(fileinfo.name));
			}
		} while (_findnext(hFile, &fileinfo) == 0);
		_findclose(hFile);
	}
}

int main(int, char** argv)
{
	int num[10];
	int j = 0;
	string s;
	int neww, newh;
	vector<string> imgfiles;
	vector<string> txtfiles;
	string imgpath = "image";
	string txtpath = "txt";
	int x1, y1, x2, y2;

	double entro,/*ssim,*/ noise;
	double entro_l,canny_l, RSS_l;
	double a = 0.8, b = 0.1;//c=
	double rlt_a, rlt_l,rlt;
	/*FILE *temp;
	temp = fopen("rlt.txt", "w");*/
	int num_image = 0;
	ofstream f("rlt.txt");
	if (!f) return 0;
	getFiles(imgpath, imgfiles);
	getFiles(txtpath, txtfiles);
	if (imgfiles.size() != 0 && txtfiles.size()!=0){
		for (vector<string>::iterator it = imgfiles.begin(), itt = txtfiles.begin(); it != imgfiles.end()&&itt!=txtfiles.end(); itt++,it++)
		{
			string img_path = *it;
			string txt_path = *itt;

			num_image++;
			ifstream infile(txt_path);
			while (infile >> s)
			{
				//cout  << s << endl;
				num[j] = atoi(s.c_str());
				//cout << num[j];
				j++;

				if (j >= 5)
				{
					j = 0;
					break;
				}
					
			}
			y1 = num[1];
			y2 = num[2];
			x1 = num[3];
			x2 = num[4];
			//cout << y1 << " " << y2 << " " << x1 << " " << x2 << " " << endl;
			// Load image
			Mat src;
			//src = imread("D:\\b.jpg");   
			src = imread(img_path);
			Mat gray;
			cvtColor(src, gray, 7);//ת���ɻҶ�ͼ 
			cout << "��"<<num_image <<"��ͼ"<< endl;
			entro = Entropy(gray)*2;  //entropy ��Ϣ��
			cout << "entropy: " << entro << "  " << endl;
			psnr = MSE_PSNR(gray);
			cout << "psnr: " << psnr << "  " << endl;
			//ssim = PSNR_SSIM(src);  //psnr �� ssim
			noise = EstimateNoise(src);
			
			rlt_a = a*entro + b*psnr + (1 - a - b)*noise;   //�������
			
			//local*************************//
			//��ȡ
			Rect rect0(x1, y1, x2 - x1, y2 - y1);
			neww = (x2 - x1)/2;
			newh = (y2 - y1)/2;
			Rect rect1(x1 - neww, y1 - newh, 4 * neww, 4 * newh);
			
			Mat image_roi0 = gray(rect0);
			Mat image_roi1 = gray(rect1);
			
			//imshow("imageROI0", image_roi0);
			/*imshow("imageROI1", image_roi1);*/
			
			imshow("origin", gray);
			

			//����
			entro_l=Entropy_local(image_roi0, image_roi1);
			RSS_l = RSS_local(image_roi0, image_roi1);
			canny_l = Canny_local(image_roi0, image_roi1);

			
			rlt_l = entro_l + RSS_l+ canny_l;

			//local*************************//

			rlt = 0.5*rlt_a + 0.5*rlt_l-7;

			//fprintf(temp, "entropy=%f,ssim=%f,psnr=%f,noise=%f,result=%rlt\n", entro,ssim,psnr,noise,rlt);
			f << "FileName=" << img_path << ",  entropy=" << entro << "	 " << /*",ssim=" << ssim << "	 " <<*/ ",psnr=" << psnr << "  " << ",noise=" << noise << " "<< ",r_a=" << rlt_a << endl;
			f << "1/entro_local: " <<  entro_l << "  " << "RSS_l:  " <<  RSS_l << "  " << "canny_l: " << canny_l << "  "<< "rlt_a: " << rlt_a<<"  " <<"rlt_l: " << rlt_l <<"  "<< "���ս��: " << rlt << endl;
			//�����ļ�
			cout << "rlt_a: "<<rlt_a << endl;
			cout << "rlt_l: " << rlt_l << endl;
			cout << "���ս��: " << rlt << endl;
	
			src.release();
		}
	}
	waitKey(0);
	return 0;

}