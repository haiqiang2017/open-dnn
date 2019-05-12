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


	Net net = readNetFromCaffe(fcn_txt,fcn_model);
	if (net.empty())
	{
		cout << "init net error";
		return -1;
	}

	float time = getTickCount(); //计时器
	cout << time;
	net.setInput(blobimg, "data");
	Mat score = net.forward("score");
	float tt = getTickCount() - time;
	cout << "time consume: " << (tt / getTickFrequency() * 1000) << endl;


	cout << score.size <<" "<< score.size[1]<<" " << score.size[2]<<" " << score.size[3] << endl;;
	const int rows = score.size[2];//height
	const int cols = score.size[3];//width
	const int chls = score.size[1];//channels

	Mat maxC1(rows, cols, CV_8UC1); //存储最终选定的通道（21分之一）
	Mat MaxVal(rows, cols, CV_32FC1);//存储label的key

	//我们要遍历的score 是数据结构为 21个通道，每个通道是500*500像素的结果
	//set 查找表
	for (int c = 0; c < chls; c++) //一共21个通道代表21个通道在每个下像素点的置信度
	{
		for (int row = 0; row < rows; row++)//500行
		{
			const float *ptrScore = score.ptr<float>(0, c, row);//指向每个通道的第1个位置，存储的是21个label的置信度 
		//	cout << *ptrScore << endl;
			uchar *ptrMaxC1 = maxC1.ptr<uchar>(row); //为了
			float *ptrMaxVal = MaxVal.ptr<float>(row);
			for(int col = 0; col < cols; col++) //500列
			{
				if (ptrScore[col] > ptrMaxVal[col])
				{
					ptrMaxVal[col] = ptrScore[col]; //存储最终的值也就是color帐的结果
					ptrMaxC1[col] = (uchar)c;//存储最终选择的通道
				}
			}
		}
	}

	//look up colors
	Mat result = Mat::zeros(rows, cols, CV_8UC3);//初始化全0的图像矩阵
	for (int row = 0; row < rows; row++)
	{
		const uchar *ptrMaxC1 = maxC1.ptr<uchar>(row); //指向对应的查找表中的行
		Vec3b *ptrColor = result.ptr<Vec3b>(row);//指向查找表中的行
		for (int col = 0; col < cols; col++)
		{
			ptrColor[col] = labels[ptrMaxC1[col]];//对应位置
		}
	}
	imshow("result", result);
	Mat dst;
	addWeighted(src, 0.4, result, 0.6, 0, dst);
	imshow("FCN", dst);

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
