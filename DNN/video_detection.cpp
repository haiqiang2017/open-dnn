#include <opencv2/opencv.hpp>
#include <opencv2/dnn.hpp>
#include <iostream>

using namespace cv;
using namespace cv::dnn;
using namespace std;

const size_t width = 300;
const size_t height = 300;
const float scaleFector = 0.007843f;
const float meanVal = 127.5;
//cv::String labelFile = "D:/new_cv/opencv/sources/samples/data/dnn/model_ssd_det/label.txt";
cv::String model_video_bin_file = "D:/new_cv/opencv/sources/samples/data/dnn/MobileNetSSD_deploy.caffemodel";
cv::String model_video_txt_file = "D:/new_cv/opencv/sources/samples/data/dnn/MobileNetSSD_deploy.prototxt";
const char* class_video_Names[] = { "background",
"aeroplane", "bicycle", "bird", "boat",
"bottle", "bus", "car", "cat", "chair",
"cow", "diningtable", "dog", "horse",
"motorbike", "person", "pottedplant",
"sheep", "sofa", "train", "tvmonitor" };

Mat detect_from_video(Mat &src)
{
	if (src.empty())
	{
		cout << "can not open file" << endl;
		exit(-1);
	}
	Net net;//初始化网络
	net = readNetFromCaffe(model_video_txt_file, model_video_bin_file);
	if (net.empty())
	{
		cout << "init the model net error";
		exit(-1);
	}
	Mat blobimg = blobFromImage(src, scaleFector, Size(300, 300), meanVal);
	net.setInput(blobimg, "data");
	Mat detection = net.forward("detection_out");//接收输出层结果
	//cout << detection.size << endl;
	//cout << detection.size[2]<<" "<< detection.size[3] << endl;
//	Mat detectionMat(detection.size[2], detection.size[3], CV_32F, detection.at<float>());//注意是ptr不是at

	Mat detectionMat(detection.size[2], detection.size[3], CV_32F, detection.ptr<float>());//声明矩阵接收结果
	const float confidence_threshold = 0.25;//定义置信度阈值
	for (int i = 0; i < detectionMat.rows; i++)
	{
		float detect_confidence = detectionMat.at<float>(i, 2);
		if (detect_confidence > confidence_threshold)//选择符合条件的结果，可能有多个圈
		{
			size_t det_index = (size_t)detectionMat.at<float>(i, 1);
			float x1 = detectionMat.at<float>(i, 3)*src.cols;
			float y1 = detectionMat.at<float>(i, 4)*src.rows;
			float x2 = detectionMat.at<float>(i, 5)*src.cols;
			float y2 = detectionMat.at<float>(i, 6)*src.rows;
			Rect rec((int)x1, (int)y1, (int)(x2 - x1), (int)(y2 - y1));
			rectangle(src,rec, Scalar(1, 1, 2), 2, 8, 0);
			putText(src, format("%s", class_video_Names[det_index]), Point(x1, y1) ,FONT_HERSHEY_SIMPLEX,1.0, Scalar(1, 2, 3), 2, 8, 0);
			imshow("src", src);
		}
	}
	return src;
}
int mainmv(int argc, char** argv)
{
	VideoCapture capture;
	capture.open("D:/new_cv/opencv/sources/samples/data/vtest.avi");
	if (!capture.isOpened())
	{
		cout << "can not open video test" << endl;
		return -1;
	}
	Mat src;
	while (capture.read(src))
	{
		detect_from_video(src);
//		imshow("src", src);

		char stp = waitKey(5);
		if (stp == 27)
		{
			break;
		}
	}
	waitKey(0);
	return 0;
}