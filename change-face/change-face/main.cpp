#include <iostream>
#include <vector>
#include <opencv2\imgproc\types_c.h>
#include <opencv2\objdetect.hpp>
#include <opencv2\features2d.hpp>
#include <opencv2\calib3d.hpp>
#include <opencv2\photo.hpp>
#include "Histogram1D.hpp"


using namespace cv;
using namespace std;

vector<Rect> DetectFace(Mat target);
Mat Change2Gray(Mat ori);
void HistSpecify(const Mat &src, const Mat &ref, Mat &result);
void EqualizeImage(const Mat &src, Mat &dst);

int main()
{
	Mat img, img_gray;
	Mat target, target_gray;
	target = imread("target.jpg");
	img = imread("ori.jpg");

	//resize(target, target, Size(img.cols, img.rows));

	//直方图规定化
	HistSpecify(target, img, target);

	//获取灰度图
	img_gray = Change2Gray(img);
	target_gray = Change2Gray(target);

	//检测两张图的脸部
	vector<Rect> faces_1 = DetectFace(target_gray);
	vector<Rect> faces_2 = DetectFace(img_gray);
	Mat target_roi;
	if (faces_1.size() == 0)
	{
		cout << "目标图没有人脸";
		getchar();
		return -1;
	}
	//Debug:脸部画矩形标记
	//rectangle(target, Point(faces_1[0].x, faces_1[0].y), Point(faces_1[0].x + faces_1[0].width, faces_1[0].y + faces_1[0].height), Scalar(0, 255, 0), 1, 8);
	Rect rect(faces_1[0].x, faces_1[0].y, faces_1[0].width, faces_1[0].height);
	if (faces_2.size() == 0)
	{
		cout << "原图没有人脸";
		getchar();
		return -1;
	}
	//Debug:脸部画矩形标记
	//rectangle(img, Point(faces_2[0].x, faces_2[0].y), Point(faces_2[0].x + faces_2[0].width, faces_2[0].y + faces_2[0].height), Scalar(0, 255, 0), 1, 8);
	//截取目标脸部
	target_roi = target(rect);
	//将目标脸部大小放缩到原图脸部大小
	resize(target_roi, target_roi, Size(faces_2[0].width, faces_2[0].height));
	//imshow("cut", target_roi);
	
	//创建遮罩
	Mat mask = Mat::zeros(img.rows, img.cols, CV_8UC3);
	Mat src = Mat::zeros(img.rows, img.cols, CV_8UC3);
	Mat white = Mat::zeros(faces_2[0].width, faces_2[0].height, CV_8UC3);
	white.setTo(Scalar(256, 256, 256));
	Rect roi_rect(faces_2[0].x, faces_2[0].y, faces_2[0].width, faces_2[0].height);
	white.copyTo(mask(roi_rect));
	target_roi.copyTo(src(roi_rect));

	//边界融合
	Mat result = Mat::zeros(img.rows, img.cols, CV_8UC3);
	Point center(faces_2[0].x + (int)(faces_2[0].width / 2), faces_2[0].y + (int)(faces_2[0].height / 2));
	seamlessClone(src, img, mask, center, result, NORMAL_CLONE);

	//高斯模糊
	GaussianBlur(img, img, Size(5, 5), 0);

	//抠出区域
	/*bitwise_not(mask, mask_turn);
	bitwise_and(img, mask_turn, img);
	target_roi.copyTo(mask(roi_rect));
	add(mask, img, img);
	GaussianBlur(img, img, Size(9, 9), 1);*/
	//imshow("mask", mask);

	int c = 0;
	
	imshow("target", target);
	imshow("ori", img);
	imshow("res", result);
	while (c != 27)
	{
		c = waitKey(1);
		if (c == 102)
		{
			flip(target_roi, target_roi, 1);
			target_roi.copyTo(src(roi_rect));
			seamlessClone(src, img, mask, center, result, NORMAL_CLONE);
			imshow("target", target);
			imshow("ori", img);
			imshow("res", result);
		
		}
	}
	imwrite("output.jpg", result);
	return 0;
}

//检测人脸位置
vector<Rect> DetectFace(Mat target)
{
	CascadeClassifier faceCascade;
	faceCascade.load("haarcascade_frontalface_alt.xml");
	vector<Rect> faces;
	faceCascade.detectMultiScale(target, faces, 1.2, 6, 0, Size(0, 0));
	return faces;
}

//获取灰度图像
Mat Change2Gray(Mat ori)
{
	Mat gray;
	if (ori.channels() == 3)
	{
		cvtColor(ori, gray, CV_BGR2GRAY);
	}
	else
	{
		gray = ori;
	}
	return gray;
}

/**
* @brief EqualizeImage 对灰度图像进行直方图均衡化
* @param src 输入图像
* @param dst 均衡化后的图像
*/
void EqualizeImage(const Mat &src, Mat &dst)
{
	Histogram1D hist1D;
	Mat hist = hist1D.getHistogram(src);

	hist /= (src.rows * src.cols); // 对得到的灰度直方图进行归一化,得到密度（0～1）
	float cdf[256] = { 0 }; // 灰度的累积概率
	Mat lut(1, 256, CV_8U); // 创建用于灰度变换的查找表
	for (int i = 0; i < 256; i++)
	{
		// 计算灰度级的累积概率
		if (i == 0)
		{
			cdf[i] = hist.at<float>(i);
		}
		else
		{
			cdf[i] = cdf[i - 1] + hist.at<float>(i);
		}
		lut.at<uchar>(i) = static_cast<uchar>(255 * cdf[i]); // 创建灰度的查找表
	}
	LUT(src, lut, dst); // 应用查找表，进行灰度变化，得到均衡化后的图像
}

/**
* @brief HistSpecify 对灰度图像进行直方图规定化
* @param src 输入图像
* @param ref 参考图像，解析参考图像的直方图并用于规定化
* @param result 直方图规定化后的图像
* @note 手动设置一个直方图并用于规定化比较麻烦，这里使用一个参考图像来进行
*/
void HistSpecify(const Mat &src, const Mat &ref, Mat &result)
{
	Histogram1D hist1D;
	Mat src_hist = hist1D.getHistogram(src);
	Mat dst_hist = hist1D.getHistogram(ref);

	float src_cdf[256] = { 0 };
	float dst_cdf[256] = { 0 };

	// 直方图进行归一化处理
	src_hist /= (src.rows * src.cols);
	dst_hist /= (ref.rows * ref.cols);

	// 计算原始直方图和规定直方图的累积概率
	for (int i = 0; i < 256; i++)
	{
		if (i == 0)
		{
			src_cdf[i] = src_hist.at<float>(i);
			dst_cdf[i] = dst_hist.at<float>(i);
		}
		else
		{
			src_cdf[i] = src_cdf[i - 1] + src_hist.at<float>(i);
			dst_cdf[i] = dst_cdf[i - 1] + dst_hist.at<float>(i);
		}
	}

	// 累积概率的差值
	float diff_cdf[256][256];
	for (int i = 0; i < 256; i++)
	{
		for (int j = 0; j < 256; j++)
		{
			diff_cdf[i][j] = fabs(src_cdf[i] - dst_cdf[j]);
		}
	}


	// 构建灰度级映射表
	Mat lut(1, 256, CV_8U);
	for (int i = 0; i < 256; i++)
	{
		// 查找源灰度级为ｉ的映射灰度
		//　和ｉ的累积概率差值最小的规定化灰度
		float min = diff_cdf[i][0];
		int index = 0;
		for (int j = 1; j < 256; j++)
		{
			if (min > diff_cdf[i][j])
			{
				min = diff_cdf[i][j];
				index = j;
			}
		}
		lut.at<uchar>(i) = static_cast<uchar>(index);
	}

	// 应用查找表，做直方图规定化
	LUT(src, lut, result);
}