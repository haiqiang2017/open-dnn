#include<iostream>
#include<opencv2/opencv.hpp>
#include<opencv2/dnn.hpp>

using namespace cv;
using namespace cv::dnn;
using namespace std;
String haar_file = "D:/new_cv/opencv/sources/data/haarcascades_cuda/haarcascade_frontalface_alt_tree.xml";
String model_age_file = "D:/new_cv/opencv/sources/data/age_net.caffemodel";
String model_age_txt = "D:/new_cv/opencv/sources/data/age_deploy.prototxt";

String model_gender_bin = "D:/new_cv/opencv/sources/data/gender_net.caffemodel";
String model_gender_txt = "D:/new_cv/opencv/sources/data/gender_deploy.prototxt";

int main(int argc, char** argv)
{
	Mat src = imread("D:/test/test.jpg");
	if (src.empty())
	{
		cout << "image empty" << endl;
		return -1;
	}
	CascadeClassifier detector;//声明级联分类器对象
	detector.load(haar_file);
	vector<Rect> faces;
	Mat gray;
	cvtColor(src, gray, COLOR_BGR2GRAY);//转成灰度图像
	detector.detectMultiScale(gray, faces, 1.02, 1, 0, Size(40, 40), Size(200, 200));

	Net age_net = readNetFromCaffe(model_age_txt, model_age_file);
	Net gender_net = readNetFromCaffe(model_gender_txt, model_gender_bin);

	for (size_t t = 0; t < faces.size(); t++)
	{
		rectangle(src, faces[t], Scalar(30, 255, 30), 2, 8, 0);
		predict_age(age_net, src(faces[t]));
		predict_gender(gender_net, src(faces[t]));
	}
	imshow("age", src);

	waitKey(0);
	return 0;
}

vector<String> agelabels()
{
	vector<String> ages;
	ages.push_back("0-2");
	ages.push_back("8 - 13");
	ages.push_back("15 - 20");
	ages.push_back("25 - 32");
	ages.push_back("38 - 43");
	ages.push_back("48 - 53");
	ages.push_back("60-");
	return ages;
}
void predict_age(Net &net, Mat &image)
{
	Mat blob = blobFromImage(image, 1.0, Size(227, 227));
	net.setInput(blob, "data");
	Mat prob = net.forward("prob");
	Mat probMat = prob.reshape(1, 1);//单通道一行
	vector<String>ages = agelabels();
	double age;
	Point index;
	minMaxLoc(probMat, NULL, &age, NULL,&index);
	int classidx = index.x;
	putText(image, format("%s", ages.at(classidx).c_str()), Point(2, 10), FONT_HERSHEY_SCRIPT_SIMPLEX, 1.0, Scalar(0, 0, 255), 2, 8, 0);
}
Mat predict_gender(Net &net, Mat &image)
{
	Mat blob = blobFromImage(image, 1.0, Size(227, 227));
	net.setInput(blob, "data");
	Mat prob = net.forward("prob");
	Mat probMat = prob.reshape(1, 1);
	Point index;
	double classnum;
	minMaxLoc(probMat, NULL, &classnum, NULL, &index);
	double classidx = index.x;
	putText(image, format("性别%s", (probMat.at<float>(0, 0) > probMat.at<float>(0, 1) ? "M":"F")), Point(2, 20), FONT_HERSHEY_SCRIPT_SIMPLEX, 1.0, Scalar(0, 0, 255), 2, 8, 0);
}