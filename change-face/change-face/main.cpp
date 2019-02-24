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

	//ֱ��ͼ�涨��
	HistSpecify(target, img, target);

	//��ȡ�Ҷ�ͼ
	img_gray = Change2Gray(img);
	target_gray = Change2Gray(target);

	//�������ͼ������
	vector<Rect> faces_1 = DetectFace(target_gray);
	vector<Rect> faces_2 = DetectFace(img_gray);
	Mat target_roi;
	if (faces_1.size() == 0)
	{
		cout << "Ŀ��ͼû������";
		getchar();
		return -1;
	}
	//Debug:���������α��
	//rectangle(target, Point(faces_1[0].x, faces_1[0].y), Point(faces_1[0].x + faces_1[0].width, faces_1[0].y + faces_1[0].height), Scalar(0, 255, 0), 1, 8);
	Rect rect(faces_1[0].x, faces_1[0].y, faces_1[0].width, faces_1[0].height);
	if (faces_2.size() == 0)
	{
		cout << "ԭͼû������";
		getchar();
		return -1;
	}
	//Debug:���������α��
	//rectangle(img, Point(faces_2[0].x, faces_2[0].y), Point(faces_2[0].x + faces_2[0].width, faces_2[0].y + faces_2[0].height), Scalar(0, 255, 0), 1, 8);
	//��ȡĿ������
	target_roi = target(rect);
	//��Ŀ��������С������ԭͼ������С
	resize(target_roi, target_roi, Size(faces_2[0].width, faces_2[0].height));
	//imshow("cut", target_roi);
	
	//��������
	Mat mask = Mat::zeros(img.rows, img.cols, CV_8UC3);
	Mat src = Mat::zeros(img.rows, img.cols, CV_8UC3);
	Mat white = Mat::zeros(faces_2[0].width, faces_2[0].height, CV_8UC3);
	white.setTo(Scalar(256, 256, 256));
	Rect roi_rect(faces_2[0].x, faces_2[0].y, faces_2[0].width, faces_2[0].height);
	white.copyTo(mask(roi_rect));
	target_roi.copyTo(src(roi_rect));

	//�߽��ں�
	Mat result = Mat::zeros(img.rows, img.cols, CV_8UC3);
	Point center(faces_2[0].x + (int)(faces_2[0].width / 2), faces_2[0].y + (int)(faces_2[0].height / 2));
	seamlessClone(src, img, mask, center, result, NORMAL_CLONE);

	//��˹ģ��
	GaussianBlur(img, img, Size(5, 5), 0);

	//�ٳ�����
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

//�������λ��
vector<Rect> DetectFace(Mat target)
{
	CascadeClassifier faceCascade;
	faceCascade.load("haarcascade_frontalface_alt.xml");
	vector<Rect> faces;
	faceCascade.detectMultiScale(target, faces, 1.2, 6, 0, Size(0, 0));
	return faces;
}

//��ȡ�Ҷ�ͼ��
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
* @brief EqualizeImage �ԻҶ�ͼ�����ֱ��ͼ���⻯
* @param src ����ͼ��
* @param dst ���⻯���ͼ��
*/
void EqualizeImage(const Mat &src, Mat &dst)
{
	Histogram1D hist1D;
	Mat hist = hist1D.getHistogram(src);

	hist /= (src.rows * src.cols); // �Եõ��ĻҶ�ֱ��ͼ���й�һ��,�õ��ܶȣ�0��1��
	float cdf[256] = { 0 }; // �Ҷȵ��ۻ�����
	Mat lut(1, 256, CV_8U); // �������ڻҶȱ任�Ĳ��ұ�
	for (int i = 0; i < 256; i++)
	{
		// ����Ҷȼ����ۻ�����
		if (i == 0)
		{
			cdf[i] = hist.at<float>(i);
		}
		else
		{
			cdf[i] = cdf[i - 1] + hist.at<float>(i);
		}
		lut.at<uchar>(i) = static_cast<uchar>(255 * cdf[i]); // �����ҶȵĲ��ұ�
	}
	LUT(src, lut, dst); // Ӧ�ò��ұ����лҶȱ仯���õ����⻯���ͼ��
}

/**
* @brief HistSpecify �ԻҶ�ͼ�����ֱ��ͼ�涨��
* @param src ����ͼ��
* @param ref �ο�ͼ�񣬽����ο�ͼ���ֱ��ͼ�����ڹ涨��
* @param result ֱ��ͼ�涨�����ͼ��
* @note �ֶ�����һ��ֱ��ͼ�����ڹ涨���Ƚ��鷳������ʹ��һ���ο�ͼ��������
*/
void HistSpecify(const Mat &src, const Mat &ref, Mat &result)
{
	Histogram1D hist1D;
	Mat src_hist = hist1D.getHistogram(src);
	Mat dst_hist = hist1D.getHistogram(ref);

	float src_cdf[256] = { 0 };
	float dst_cdf[256] = { 0 };

	// ֱ��ͼ���й�һ������
	src_hist /= (src.rows * src.cols);
	dst_hist /= (ref.rows * ref.cols);

	// ����ԭʼֱ��ͼ�͹涨ֱ��ͼ���ۻ�����
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

	// �ۻ����ʵĲ�ֵ
	float diff_cdf[256][256];
	for (int i = 0; i < 256; i++)
	{
		for (int j = 0; j < 256; j++)
		{
			diff_cdf[i][j] = fabs(src_cdf[i] - dst_cdf[j]);
		}
	}


	// �����Ҷȼ�ӳ���
	Mat lut(1, 256, CV_8U);
	for (int i = 0; i < 256; i++)
	{
		// ����Դ�Ҷȼ�Ϊ���ӳ��Ҷ�
		//���ͣ���ۻ����ʲ�ֵ��С�Ĺ涨���Ҷ�
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

	// Ӧ�ò��ұ���ֱ��ͼ�涨��
	LUT(src, lut, result);
}