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
cv::String model_bin_file = "D:/new_cv/opencv/sources/samples/data/dnn/MobileNetSSD_deploy.caffemodel";
cv::String model_txt_file = "D:/new_cv/opencv/sources/samples/data/dnn/MobileNetSSD_deploy.prototxt";
const char* classNames[] = { "background",
"aeroplane", "bicycle", "bird", "boat",
"bottle", "bus", "car", "cat", "chair",
"cow", "diningtable", "dog", "horse",
"motorbike", "person", "pottedplant",
"sheep", "sofa", "train", "tvmonitor" };

const int meanValues[3] = { 104, 117, 123 };


int mainm(int argc, char** argv) {
	Mat frame = imread("D:/test/test.jpg");
	if (frame.empty()) {
		printf("could not load image...\n");
		return -1;
	}
	namedWindow("input image", CV_WINDOW_AUTOSIZE);
	imshow("input image", frame);

	Net net = readNetFromCaffe(model_txt_file, model_bin_file);

	Mat blobImage = blobFromImage(frame,scaleFector,Size(300,300),Scalar(104, 117, 123),false);
	//imshow("blobImage", blobImage);
	net.setInput(blobImage, "data");
	Mat detection = net.forward("detection_out");

	Mat detectionMat(detection.size[2], detection.size[3], CV_32F, detection.ptr<float>());
	float confidence_threshold = 0.6;
	cout << "rows" << detectionMat.rows << endl;
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
			putText(frame, format("%s", classNames[objIndex]), Point(20, 20), FONT_HERSHEY_COMPLEX, 1.0, Scalar(3, 9, 45), 2, 8, 0);
		}
		imshow("ssd-demo", frame);

		waitKey(0);
		return 0;
	}
}
