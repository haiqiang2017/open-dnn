#include<opencv2/opencv.hpp>
#include<opencv2/dnn.hpp>
#include<iostream>
using namespace std;
using namespace cv::dnn;
using namespace dnn;
using namespace cv;


const size_t width = 300;
const size_t height = 300;
const int meanValues[3] = { 104,117,123 };
cv::String model_det_bin_file = "D:/new_cv/opencv/sources/samples/data/dnn/model_ssd_det/VGG_ILSVRC2016_SSD_300x300_iter_440000.caffemodel";
cv::String model_det_txt_file = "D:/new_cv/opencv/sources/samples/data/dnn/model_ssd_det/deploy.prototxt";
cv::String label_file = "";
const char* classNames[] = { "background",
"aeroplane", "bicycle", "bird", "boat",
"bottle", "bus", "car", "cat", "chair",
"cow", "diningtable", "dog", "horse",
"motorbike", "person", "pottedplant",
"sheep", "sofa", "train", "tvmonitor" };

vector<String> readLabelsdet();
static Mat getMean(const size_t &w, const size_t &h)
{
	Mat mean;
	vector<Mat> channels;
	for (int i = 0; i < 3; i++)
	{
		Mat channel(h, w, CV_32F, Scalar(meanValues[i]));
		channels.push_back(channel);
	}
	merge(channels, mean);
	return mean;
}

static Mat propress(const Mat & frame)
{
	Mat propressed;
	frame.convertTo(propressed, CV_32F);
	resize(propressed, propressed, Size(width, height));
	Mat mean = getMean(width, height);
	subtract(propressed, mean, propressed);//与均值做差，并返回结果
	return propressed;
}
int main(int argc, char** argv)
{
	Mat frame = imread("D:/test/test.jpg");
	if (frame.empty())
	{
		cout << "img empty" << endl;
		return -1;
	}
	imshow("frame", frame);
	
	Ptr <dnn::Importer>importer;
	try {
		importer = createCaffeImporter(model_det_txt_file, model_det_bin_file);
	}
	catch (const cv::Exception &err)
	{
		cerr << err.msg << endl;
	}
	Net net;
	importer->populateNet(net);//填充
	importer.release();

	Mat input_frame = propress(frame);
	Mat blobImage = blobFromImage(input_frame);
	net.setInput(blobImage, "data");
	Mat detection = net.forward("detection_out");
	Mat detectionMat(detection.size[2], detection.size[3], CV_32F, detection.ptr<float>());
	float confidenc_threshold = 0.2;
	for (int i = 0; i < detectionMat.rows; i++)
	{
		float confluence = detectionMat.at<float>(i,2);
		if (confluence > confidenc_threshold)
		{
			size_t objIndex = (size_t)(detectionMat.at<float>(i, 1));
			float t1_x = detectionMat.at<float>(i, 3)*frame.cols;
			float t1_y = detectionMat.at<float>(i, 4)*frame.rows;
			float br_x = detectionMat.at<float>(i, 5)*frame.cols;
			float br_y = detectionMat.at<float>(i, 6)*frame.rows;

			Rect object_box((int)t1_x, (int)t1_y, (int)(br_x - t1_x), (int)(br_y - t1_y);
			rectangle(frame, object_box, Scalar(0, 0, 255), 2, 8, 0);
			putText(frame, format("%s", objNames[objIndex].c_str()), Point(tl_x, tl_y), FONT_HERSHEY_SIMPLEX, 1.0, Scalar(255, 0, 0), 2);
		}
	}

	waitKey(0);
	return 0;
}
vector<String>readLabelsdet()
{
	vector<String>result;
	ifstream fp(label_file);
	if (!fp.is_open())
	{
		cout << "can not open label file" << endl;
		exit(-1);
	}
	string name;
	while  (!fp.eof())
	{
		getline(fp, name);
		if (name.length() && name.find("display_name") == 0)
		{
			string tmp = name.substr(15);
			tmp.replace(tmp.end() - 1, tmp.end(), "");
			result.push_back(name);
		}
	}
	fp.close();
	return result;
}