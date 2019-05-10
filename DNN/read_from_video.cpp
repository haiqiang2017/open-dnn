#include <opencv2/opencv.hpp>
#include<iostream>
using namespace std;
using namespace cv;

int main(int argc, char** argv)
{

	VideoCapture capture;
	capture.open("D:/new_cv/opencv/sources/samples/data/vtest.avi");
	if (!capture.isOpened())
	{
		cout << "can't open video" << endl;
		return -1;
	}
	Mat frame;
	Mat bsmaskMOG2, bsmaskKNN;
	namedWindow("input", CV_WINDOW_AUTOSIZE);
	namedWindow("MOG2", CV_WINDOW_AUTOSIZE);
	Mat kernel = getStructuringElement(MORPH_RECT, Size(3, 3), Point(-1, -1));
	Ptr<BackgroundSubtractor>pMOG2 = createBackgroundSubtractorMOG2();
	while (capture.read(frame))
	{
		imshow("input", frame);
		pMOG2->apply(frame, bsmaskMOG2);
		morphologyEx(bsmaskMOG2, bsmaskMOG2, MORPH_OPEN, kernel, Point(-1, -1));
		imshow("MOG2", bsmaskMOG2);
		char c = waitKey(100);
		if (c == 27)
		{
			break;
		}
	}
	waitKey(0);
	return 0;
}