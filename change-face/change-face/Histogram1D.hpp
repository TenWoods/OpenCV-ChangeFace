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
	int histSize[1]; // 项的数量
	float hranges[2]; // 统计像素的最大值和最小值
	const float* ranges[1];
	int channels[1]; // 仅计算一个通道

public:
	Histogram1D()
	{
		// 准备1D直方图的参数
		histSize[0] = 256;
		hranges[0] = 0.0f;
		hranges[1] = 255.0f;
		ranges[0] = hranges;
		channels[0] = 0;
	}

	Mat getHistogram(const Mat &image)
	{
		Mat hist;
		// 计算直方图
		calcHist(&image,// 要计算图像的
			1,                // 只计算一幅图像的直方图
			channels,        // 通道数量
			Mat(),            // 不使用掩码
			hist,            // 存放直方图
			1,                // 1D直方图
			histSize,        // 统计的灰度的个数
			ranges);        // 灰度值的范围
		return hist;
	}

	Mat getHistogramImage(const Mat &image)
	{
		Mat hist = getHistogram(image);

		//查找最大值用于归一化
		double maxVal = 0;

		minMaxLoc(hist, NULL, &maxVal);

		//绘制直方图的图像
		Mat histImg(histSize[0], histSize[0], CV_8U, Scalar(255));

		// 设置最高点为最大值的90%
		double hpt = 0.9 * histSize[0];
		//每个条目绘制一条垂直线
		for (int h = 0; h < histSize[0]; h++)
		{
			//直方图的元素类型为32位浮点数
			float binVal = hist.at<float>(h);
			int intensity = static_cast<int>(binVal * hpt / maxVal);
			line(histImg, Point(h, histSize[0]),
				Point(h, histSize[0] - intensity), Scalar::all(0));
		}
		return histImg;
	}
};