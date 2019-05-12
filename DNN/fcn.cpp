#include<iostream>	
#include<opencv2/opencv.hpp>
#include<opencv2/dnn.hpp>
using namespace std;
using namespace cv;
using namespace cv::dnn;

String fcn_label_txt = "D:/new_cv/opencv/sources/samples/data/dnn/pascal-classes.txt";
String fcn_txt = "D:/new_cv/opencv/sources/samples/data/dnn/fcn8s-heavy-pascal.prototxt";
String fcn_model = "D:/new_cv/opencv/sources/samples/data/dnn/fcn8s-heavy-pascal.caffemodel";

vector<Vec3b> readColors();
int main(int argc, char** argv)
{

	Mat src = imread("D:/test/test.jpg");
	if (src.empty())
	{
		cout << "imput img is empty" << endl;
		return -1;
	}
	//resize
	resize(src, src, Size(500, 500));
	imshow("src", src);
	vector<Vec3b> labels = readColors();
	Mat blobimg = blobFromImage(src);

	//init net
	Net net = readNetFromCaffe(fcn_txt,fcn_model);
	if (net.empty())
	{
		cout << "init net error";
		return -1;
	}

	//use net
	float time = getTickCount();
	cout << time;
	net.setInput(blobimg, "data");
	Mat score = net.forward("score");
	float tt = getTickCount() - time;
	cout << "time consume: " << (tt / getTickFrequency() * 1000) << endl;

	const int rows = score.size[2];//height
	const int cols = score.size[3];//width

	waitKey(0);
	return 0;
}
vector<Vec3b> readColors()
{
	vector<Vec3b> colors;
	ifstream fp(fcn_label_txt);//创建流
	if (!fp.is_open())
	{
		cout << "can not open label file" << endl;
		exit(-1);
	}
	string line;
	while (!fp.eof())//如果流没有结束
	{
		getline(fp, line);
		if (line.length())//判断是否可以打开
		{
			stringstream ss(line);//创建字符流
			string name;
			ss >> name;
		//	cout << name << endl;;
			int temp;
			Vec3b color;
			ss >> temp;
			cout << temp;
			color[0] = (uchar)temp;
			ss >> temp;
			color[1] = (uchar)temp;
			ss >> temp;
			color[2] = (uchar)temp;
	//		cout << temp << endl;
			colors.push_back(color);
		}
	}
	return colors;
}
/*
background 0 0 0
aeroplane 128 0 0
bicycle 0 128 0
bird 128 128 0
*/
