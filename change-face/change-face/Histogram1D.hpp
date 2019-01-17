#include <iostream>
#include <vector>
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/features2d.hpp>
#include <opencv2/calib3d.hpp>

using namespace std;
using namespace cv;

class Histogram1D
{
private:
	int histSize[1]; // �������
	float hranges[2]; // ͳ�����ص����ֵ����Сֵ
	const float* ranges[1];
	int channels[1]; // ������һ��ͨ��

public:
	Histogram1D()
	{
		// ׼��1Dֱ��ͼ�Ĳ���
		histSize[0] = 256;
		hranges[0] = 0.0f;
		hranges[1] = 255.0f;
		ranges[0] = hranges;
		channels[0] = 0;
	}

	Mat getHistogram(const Mat &image)
	{
		Mat hist;
		// ����ֱ��ͼ
		calcHist(&image,// Ҫ����ͼ���
			1,                // ֻ����һ��ͼ���ֱ��ͼ
			channels,        // ͨ������
			Mat(),            // ��ʹ������
			hist,            // ���ֱ��ͼ
			1,                // 1Dֱ��ͼ
			histSize,        // ͳ�ƵĻҶȵĸ���
			ranges);        // �Ҷ�ֵ�ķ�Χ
		return hist;
	}

	Mat getHistogramImage(const Mat &image)
	{
		Mat hist = getHistogram(image);

		//�������ֵ���ڹ�һ��
		double maxVal = 0;

		minMaxLoc(hist, NULL, &maxVal);

		//����ֱ��ͼ��ͼ��
		Mat histImg(histSize[0], histSize[0], CV_8U, Scalar(255));

		// ������ߵ�Ϊ���ֵ��90%
		double hpt = 0.9 * histSize[0];
		//ÿ����Ŀ����һ����ֱ��
		for (int h = 0; h < histSize[0]; h++)
		{
			//ֱ��ͼ��Ԫ������Ϊ32λ������
			float binVal = hist.at<float>(h);
			int intensity = static_cast<int>(binVal * hpt / maxVal);
			line(histImg, Point(h, histSize[0]),
				Point(h, histSize[0] - intensity), Scalar::all(0));
		}
		return histImg;
	}
};