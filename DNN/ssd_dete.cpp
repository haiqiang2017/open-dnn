#include <opencv2/opencv.hpp>
#include <opencv2/dnn.hpp>
#include <iostream>

using namespace cv;
using namespace cv::dnn;
using namespace std;

const size_t width = 300;
const size_t height = 300;
cv::String labelFile = "D:/new_cv/opencv/sources/samples/data/dnn/model_ssd_det/label.txt";
cv::String model_det_bin_file = "D:/new_cv/opencv/sources/samples/data/dnn/model_ssd_det/VGG_ILSVRC2016_SSD_300x300_iter_440000.caffemodel";
cv::String model_det_txt_file = "D:/new_cv/opencv/sources/samples/data/dnn/model_ssd_det/deploy.prototxt";
const char* classNames[] = { "background",
"aeroplane", "bicycle", "bird", "boat",
"bottle", "bus", "car", "cat", "chair",
"cow", "diningtable", "dog", "horse",
"motorbike", "person", "pottedplant",
"sheep", "sofa", "train", "tvmonitor" };
vector<String> readLabels();
vector<String> readlabel();
vector<String> readclass();
const int meanValues[3] = { 104, 117, 123 };
static Mat getMean(const size_t &w, const size_t &h) {
	Mat mean;
	vector<Mat> channels;
	for (int i = 0; i < 3; i++) {
		Mat channel(h, w, CV_32F, Scalar(meanValues[i]));
		channels.push_back(channel);
	}
	merge(channels, mean);//把通道放到mean中 split是分割图像
	
	return mean;
}

static Mat preprocess(const Mat &frame) {
	Mat preprocessed;
	frame.convertTo(preprocessed, CV_32F);
	resize(preprocessed, preprocessed, Size(width, height)); // 300x300 image
	Mat mean = getMean(width, height);//获取均值图像
	imshow("mean", mean);
	subtract(preprocessed, mean, preprocessed);//原图像减去均值图像，获取主要特征
	imshow("preprocessed", preprocessed);
	return preprocessed;
}

int main1(int argc, char** argv) {
	Mat frame = imread("D:/new_cv/opencv/sources/samples/data/lena.jpg");
	if (frame.empty()) {
		printf("could not load image...\n");
		return -1;
	}
	namedWindow("input image", CV_WINDOW_AUTOSIZE);
	imshow("input image", frame);

	vector<String> objNames = readlabel();
	//vector<String> objNames = readclass();
	
	// import Caffe SSD model
	//Ptr<dnn::Importer> importer;
	//try {
	//	importer = createCaffeImporter(model_det_txt_file, model_det_bin_file);
	//}
	//catch (const cv::Exception &err) {
	//	cerr << err.msg << endl;
	//}
	//Net net;
	//importer->populateNet(net);
	//importer.release();

	Net net = readNetFromCaffe(model_det_txt_file, model_det_bin_file);
	Mat input_image = preprocess(frame);
	Mat blobImage = blobFromImage(input_image);
	//imshow("blobImage", blobImage);
	net.setInput(blobImage, "data");
	Mat detection = net.forward("detection_out");
	cout << detection.size[2] << detection.size[3] << endl;

	Mat detectionMat(detection.size[2], detection.size[3], CV_32F, detection.ptr<float>());
	float confidence_threshold = 0.2;
	cout <<"rows" <<detectionMat.rows << endl;
	cout << "cols" << detectionMat.cols << endl;
	for (int i = 0; i < detectionMat.rows; i++) {
		float confidence = detectionMat.at<float>(i, 2);
		if (confidence > confidence_threshold) {
			size_t objIndex = (size_t)(detectionMat.at<float>(i, 1));
	//		cout << objIndex << endl;
			float tl_x = detectionMat.at<float>(i, 3) * frame.cols;
			float tl_y = detectionMat.at<float>(i, 4) * frame.rows;
			float br_x = detectionMat.at<float>(i, 5) * frame.cols;
			float br_y = detectionMat.at<float>(i, 6) * frame.rows;

			Rect object_box((int)tl_x, (int)tl_y, (int)(br_x - tl_x), (int)(br_y - tl_y));
			rectangle(frame, object_box, Scalar(0, 0, 255), 2, 8, 0);
	//		cout << objNames[objIndex].c_str() << endl;
	//		putText(frame, format("%s", objNames[objIndex].c_str()),Point(tl_x, tl_y), FONT_HERSHEY_PLAIN, 1.0, Scalar(255, 0, 0), 2);
			/*
			原型 void putText( Mat& img, const string& text, Point org, int fontFace,double fontScale，  Scalar color, int thickness=1, int lineType=8 );

            参数1：， Mat& img，待写字的图片，我们写在img图上

			参数2：，const string& text，待写入的字，我们下面写入Hello
			
			参数3：， Point org， 第一个字符左下角坐标，我们设定在图片的Point（50,60）坐标。表示x = 50,y = 60。
			
			参数4：，int fontFace，字体类型，FONT_HERSHEY_SIMPLEX ，FONT_HERSHEY_PLAIN ，FONT_HERSHEY_DUPLEX 等等等。
			
			参数5：，double fontScale，字体大小，我们设置为2号
			
			参数6：，Scalar color，字体颜色，颜色用Scalar（）表示
			

			参数7：， int thickness，字体粗细，我们下面代码使用的是4号
			
			参数8：， int lineType，线型，我们使用默认值8.
			*/
		}
	}
	imshow("ssd-demo", frame);

	waitKey(0);
	return 0;
}
//cvtColor 色彩空间转换函数
vector<String> readclass()
{
	vector<String> result;
	for (int i = 0; i < sizeof(classNames); i++)
	{
		result.push_back(string(classNames[i]));
	}
	return result;
}
vector<String> readlabel()
{
	ifstream fp(labelFile);
	cout << labelFile;
	vector<String> result;
	if (!fp.is_open())
	{
		cout << "open file error" << endl;
		exit(-1);
	}
	string name;
	while (!fp.eof())
	{
		getline(fp,name);
		if (name.length())
		{
			string d1 = name.substr(name.find(",") + 1);//第一个，的后面
			string d2 = d1.substr(d1.find(",") + 1);
			result.push_back(d2);
		}
	}
	for (vector<String>::iterator it = result.begin(); it != result.end(); it++)
	{
		cout << *it << endl;
	}
	return result;
}

vector<String> readLabels() {
	vector<String> objNames;
	ifstream fp(labelFile);
	if (!fp.is_open()) {
		printf("could not open the file...\n");
		exit(-1);
	}
	string name;
	while (!fp.eof()) {
		getline(fp, name);
		if (name.length() && (name.find("display_name:") == 0)) {
			string temp = name.substr(15);
			temp.replace(temp.end() - 1, temp.end(), "");
			objNames.push_back(temp);
		}
	}
	return objNames;
}

