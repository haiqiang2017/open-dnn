#include <opencv2/opencv.hpp>
#include<iostream>
#include<opencv2/dnn.hpp>
using namespace std;
using namespace cv;
using namespace cv::dnn;
void predict_age(Mat &src);
void predict_gender(Mat &src);
String case_file = "D:/new_cv/opencv/sources/data/haarcascades/haarcascade_frontalface_alt_tree.xml";
String model_age1_file = "D:/new_cv/opencv/sources/samples/data/age_net.caffemodel";
String model_age1_txt = "D:/new_cv/opencv/sources/samples/data/age_deploy.prototxt";

String model_gender1_bin = "D:/new_cv/opencv/sources/samples/data/gender_net.caffemodel";
String model_gender1_txt = "D:/new_cv/opencv/sources/samples/data/gender_deploy.prototxt";

int main(int argc, char** argv)
{


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
	cascader.detectMultiScale(gray, res, 1.02, 1, 0, Size(27, 27));
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
	cout << res.size() << endl;
	for (size_t t = 0; t < res.size(); t++)
	{
		rectangle(src, res[t], Scalar(1, 1, 2), 1, 8, 0);
		Mat dst = src(res[t]);
		predict_age(dst);
		predict_gender(dst);
	}
	imshow("detection result", src);

	waitKey(0);
	return 0;
}
vector<String> get_age_label()
{
	vector<String> age_labels;
	age_labels.push_back("0-2");
	age_labels.push_back("4-6");
	age_labels.push_back("8-13");
	age_labels.push_back("15-20");
	age_labels.push_back("25-32");
	age_labels.push_back("38-43");
	age_labels.push_back("48-53");
	age_labels.push_back("60-");
	return age_labels;
}
void predict_age(Mat &src)
{
	Net net = readNetFromCaffe(model_age1_txt, model_age1_file);
	if (net.empty())
	{
		cout << "load net error" << endl;
		exit(-1);
	}
	Mat blobImg = blobFromImage(src, 1.0, Size(227, 227));
	net.setInput(blobImg, "data");
	Mat probMat = net.forward("prob");
	probMat.reshape(1, 1);//1行1通道
	Point index;//坐标信息
	double objvalue;//最大检测值
	minMaxLoc(probMat, NULL, &objvalue, NULL, &index);//忽略最小值，取最大值
	size_t objindex = index.x;//label 下标
	vector<String>labels = get_age_label();
	putText(src, format("age:%s", labels[objindex].c_str()), Point(2, 20), FONT_HERSHEY_PLAIN, 0.7, Scalar(1, 12, 3), 1, 8);
}
void predict_gender(Mat &src)
{
	Net net = readNetFromCaffe(model_age1_txt, model_age1_file);
	if (net.empty())
	{
		cout << "load net error" << endl;
		exit(-1);
	}
	Mat blobImg = blobFromImage(src, 1.0, Size(227, 227));
	net.setInput(blobImg, "data");
	Mat probMat = net.forward("prob");
	probMat.reshape(1, 1);//1行1通道
	putText(src, format("gender:%s", (probMat.size[0] > probMat.size[1] ? "M" : "F")), Point(2, 10), FONT_HERSHEY_PLAIN, 0.7, Scalar(2, 2, 3), 1, 8);
}