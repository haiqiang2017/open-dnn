#include<stdio.h>
#include<opencv2/opencv.hpp>
#include<opencv2/dnn.hpp>
using namespace std;
using namespace cv;
using namespace cv::dnn;


cv::String model_gbin_file = "D:/new_cv/opencv/sources/samples/data/dnn/bvlc_googlenet.prototxt";
cv::String model_gtxt_file = "D:/new_cv/opencv/sources/samples/data/dnn/bvlc_googlenet.caffemodel";
//cv::String model_gbin_file = "D:/C++/opencv/class7model/MobileNetSSD_deploy.caffemodel";
//cv::String model_gtxt_file = "D:/C++/opencv/class7model/MobileNetSSD_deploy.prototxt";
String label_gtxt_file = "D:/new_cv/opencv/sources/samples/data/dnn/synset_words.txt";
vector<String> readlabels();

int maing(int argc, char** argv)
{
	Mat src = imread("D:/new_cv/opencv/sources/samples/data/apple.jpg");
	if (src.empty())
	{
		cout << "load image error" << endl;
		return -1;	
	}
	namedWindow("src", CV_WINDOW_AUTOSIZE);
	imshow("src", src);
	vector<String>labels = readlabels();
	Net net = readNetFromCaffe(model_gbin_file,model_gtxt_file);
	if (net.empty())
	{
		cout << "error net" << endl;
		return -1;
	}

	Mat inputBlob = blobFromImage(src,1.0,Size(224,224),Scalar(104,117,123));//原始图像，是否放缩，Size尺寸,均值means	
	Mat prob;
	for (int i = 0; i < 10; i++)
	{
		net.setInput(inputBlob, "data");//输入plob和层的类型
	    prob = net.forward("prob");
	}
	Mat probMat = prob.reshape(1, 1);//最大可能性的位置
	Point classNumber;

	double classProb;
	//minMaxLoc(probMat, NULL, &classProb, NULL, &classNumber);
	minMaxLoc(probMat, NULL, &classProb, NULL, &classNumber);

	int classidx = classNumber.x;
	cout << "current classfiction is" << labels.at(classidx).c_str() << classProb << endl;

	putText(src, labels.at(classidx).c_str(), Point(20, 20), FONT_HERSHEY_SIMPLEX ,1.0, Scalar(0, 0, 255), 2, 8);
	imshow("new image", src);
	waitKey(0);
	return 0;
}
vector<String>readlabels()
{
	vector<String> classNames;
	ifstream fp(label_gtxt_file);
	if (!fp.is_open())
	{
		cout << "can not open label txt" << endl;
		exit(-1);
	}
	string name;
	while (!fp.eof())
	{
		getline(fp, name);
		if (name.length())
		{
			classNames.push_back(name.substr(name.find(' ') + 1));
		}
	}
	fp.close();
	return classNames;
}