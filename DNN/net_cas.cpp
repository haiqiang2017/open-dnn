#include <opencv2/opencv.hpp>
#include<iostream>
#include<opencv2/dnn.hpp>
using namespace std;
using namespace cv;
using namespace cv::dnn;

int main(int argc, char** argv)
{
	String case_file = "D:/new_cv/opencv/sources/data/haarcascades/haarcascade_frontalface_alt_tree.xml";
	Mat src = imread("D:/test/test.jpg");
	if (src.empty())
	{
		cout << "src is empty" << endl;
		return -1;
	}

	CascadeClassifier cascader;//声明级联对象

	cascader.load(case_file);
	if (cascader.empty())
	{
		cout << "load error" << endl;
		return -1;
	}
	vector<Rect> res;//级联返回的是坐标对象
	Mat gray;//使用灰度图像进行检测
	cvtColor(src, gray, CV_RGB2GRAY);//获取灰度图像
	imshow("gray", gray);
	equalizeHist(gray, gray);
	cascader.detectMultiScale(gray, res, 1.2, 3, 0, Size(27, 27));
	/*
	void detectMultiScale(
	const Mat& image,                //待检测灰度图像
	CV_OUT vector<Rect>& objects,    //被检测物体的矩形框向量
	double scaleFactor = 1.1,        //前后两次相继的扫描中搜索窗口的比例系数，默认为1.1 即每次搜索窗口扩大10%
	int minNeighbors = 3,            //构成检测目标的相邻矩形的最小个数 如果组成检测目标的小矩形的个数和小于minneighbors - 1 都会被排除
									 //如果minneighbors为0 则函数不做任何操作就返回所有被检候选矩形框
	int flags = 0,                   //若设置为CV_HAAR_DO_CANNY_PRUNING 函数将会使用Canny边缘检测来排除边缘过多或过少的区域
	Size minSize = Size(),
	Size maxSize = Size()            //最后两个参数用来限制得到的目标区域的范围
	);
	*/
	for (size_t t = 0; t < res.size(); t++)
	{
		rectangle(src, res[t], Scalar(1, 1, 2), 1, 8, 0);
	}
	imshow("detection result", src);

	waitKey(0);
	return 0;
}