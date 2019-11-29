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

//LUMINANCE參數
double mean_as;//sigma image & shift_image
double mean_a;
double mean_s;
double luminance;
//CONTRAST參數
double contrast_a;
double contrast_s;
double contrast;*/



/**PSNR+SSIM SSIM 越小越复杂 PSNR**/
double MSE_PSNR(Mat image)
{//PSNR 数值越大图像失真越小，图像质量越高，则近似为复杂度越低
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
	psnr = 1 / psnr+6;		//复杂度归一化		
	
	return psnr;
}

/*double L(int rows, int cols, Mat image, Mat shift_image)
{//亮度
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
{//对比度
	double variance_a = 0;//image變異數
	double variance_s = 0;//shift_image變異數
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
{//结构
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
	//SSIM取值范围[0,1]，值越大，表示图像失真越小.质量越高，近似为复杂度低
	int cols = image.cols;//width
	int rows = image.rows;//height
	double l;
	double c;
	double s;
	double ssim;
	Mat shift_image = Mat(rows, cols, CV_8UC1);
	for (j = 0; j<cols - 1; j++)//右移
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
	ssim = 100 / ssim - 1;//复杂度数值归一化
	cout << "ssim: " << ssim << endl;
	cout << "psnr: " << psnr << endl;

	return ssim;
}*/    //ssim 暂时停用

/**噪音估计 噪音越大越复杂**/
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

/**Engropy 信息熵 信息熵越大越复杂**/
double Entropy(Mat img)
{
	// 将输入的矩阵为图像
	double temp[256];

	double num[256];
	int sum = 0;
	double var = 0;
	double aver = 0;

	// 清零
	for (int i = 0; i<256; i++)
	{
		temp[i] = 0.0;
		num[i] = 0;
	}

	// 计算每个像素的累积值
	for (int m = 0; m<img.rows; m++)
	{// 有效访问行列的方式
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

	// 计算每个像素的概率
	for (int i = 0; i<256; i++)
	{
		temp[i] = temp[i] / (img.rows*img.cols);
	}
	double result1 = 0;
	double result2 = 0;    //加权信息熵
	double result3 = 0;    //方差加权信息熵
	// 根据定义计算图像熵
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
			result1 = result1 - temp[i] * (log(temp[i]) / log(2.0));   //信息熵
			result2 = result2 - i*temp[i] * (log(temp[i]) / log(2.0));   //加权信息熵
			result3 = result3 - var*temp[i] * (log(temp[i]) / log(2.0));  //方差加权信息熵
		}
	}

	//cout << "信息熵: " << result1 << endl;
	/*cout << "加权信息熵: " << result2 << endl;
	cout << "方差信息熵: " << result3 << endl;*/
	
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
	//目标与限定邻近区域的边缘比率  值越大越好提取目标，复杂度越低
	Mat  edge;
	edge.create(img.size(), img.type());	// 创建与src同类型和大小的矩阵
											// 运行Canny算子  
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
	//显示效果图   
	imshow("Canny边缘检测", edge);

	canny_p = 1 / canny_p - 20;
	canny_p = 10 / (1 + exp(-canny_p + 1));

	cout << "canny_l: " << canny_p<< endl;
	return canny_p;//边缘比率
}

void getFiles(string path, vector<string>&files){
	//文件句柄  
	long   hFile = 0;
	//文件信息  
	struct _finddata_t fileinfo;
	string p;
	if ((hFile = _findfirst(p.assign(path).append("\\*").c_str(), &fileinfo)) != -1)
	{
		do
		{
			//如果是目录,迭代之  
			//如果不是,加入列表  
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
			cvtColor(src, gray, 7);//转换成灰度图 
			cout << "第"<<num_image <<"张图"<< endl;
			entro = Entropy(gray)*2;  //entropy 信息熵
			cout << "entropy: " << entro << "  " << endl;
			psnr = MSE_PSNR(gray);
			cout << "psnr: " << psnr << "  " << endl;
			//ssim = PSNR_SSIM(src);  //psnr 和 ssim
			noise = EstimateNoise(src);
			
			rlt_a = a*entro + b*psnr + (1 - a - b)*noise;   //结果计算
			
			//local*************************//
			//截取
			Rect rect0(x1, y1, x2 - x1, y2 - y1);
			neww = (x2 - x1)/2;
			newh = (y2 - y1)/2;
			Rect rect1(x1 - neww, y1 - newh, 4 * neww, 4 * newh);
			
			Mat image_roi0 = gray(rect0);
			Mat image_roi1 = gray(rect1);
			
			//imshow("imageROI0", image_roi0);
			/*imshow("imageROI1", image_roi1);*/
			
			imshow("origin", gray);
			

			//量化
			entro_l=Entropy_local(image_roi0, image_roi1);
			RSS_l = RSS_local(image_roi0, image_roi1);
			canny_l = Canny_local(image_roi0, image_roi1);

			
			rlt_l = entro_l + RSS_l+ canny_l;

			//local*************************//

			rlt = 0.5*rlt_a + 0.5*rlt_l-7;

			//fprintf(temp, "entropy=%f,ssim=%f,psnr=%f,noise=%f,result=%rlt\n", entro,ssim,psnr,noise,rlt);
			f << "FileName=" << img_path << ",  entropy=" << entro << "	 " << /*",ssim=" << ssim << "	 " <<*/ ",psnr=" << psnr << "  " << ",noise=" << noise << " "<< ",r_a=" << rlt_a << endl;
			f << "1/entro_local: " <<  entro_l << "  " << "RSS_l:  " <<  RSS_l << "  " << "canny_l: " << canny_l << "  "<< "rlt_a: " << rlt_a<<"  " <<"rlt_l: " << rlt_l <<"  "<< "最终结果: " << rlt << endl;
			//存入文件
			cout << "rlt_a: "<<rlt_a << endl;
			cout << "rlt_l: " << rlt_l << endl;
			cout << "最终结果: " << rlt << endl;
	
			src.release();
		}
	}
	waitKey(0);
	return 0;

}