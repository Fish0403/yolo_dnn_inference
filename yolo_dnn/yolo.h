#pragma once
#include<iostream>
#include<opencv2/opencv.hpp>
using namespace std;
using namespace cv;
using namespace cv::dnn;
struct Detection
{
	int class_id{ 0 };//结果类别id
	float confidence{ 0.0 };//结果置信度
	cv::Rect box{};//矩形框
};
class Yolo {
	public:
		bool readModel(cv::dnn::Net& net, std::string& netPath, bool isCuda);
		void drawPred(cv::Mat& img, std::vector<Detection> result, std::vector<cv::Scalar> color);
		virtual	vector<Detection> Detect(cv::Mat& SrcImg, cv::dnn::Net& net) = 0;
		float sigmoid_x(float x) { return static_cast<float>(1.f / (1.f + exp(-x))); }
		Mat formatToSquare(const cv::Mat& source)
		{
			int col = source.cols;
			int row = source.rows;
			int _max = MAX(col, row);
			cv::Mat result = cv::Mat::zeros(_max, _max, CV_8UC3);
			source.copyTo(result(cv::Rect(0, 0, col, row)));
			return result;
		}
		const int netWidth = 640;   //ONNX图片输入宽度
		const int netHeight = 640;  //ONNX图片输入高度


		float modelConfidenceThreshold{ 0.0 };
		float modelScoreThreshold{ 0.0 };
		float modelNMSThreshold{ 0.0 };

		std::vector<std::string> classes = { "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat", "traffic light",
			"fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep", "cow",
			"elephant", "bear", "zebra", "giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee",
			"skis", "snowboard", "sports ball", "kite", "baseball bat", "baseball glove", "skateboard", "surfboard",
			"tennis racket", "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple",
			"sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "couch",
			"potted plant", "bed", "dining table", "toilet", "tv", "laptop", "mouse", "remote", "keyboard", "cell phone",
			"microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors", "teddy bear",
			"hair drier", "toothbrush" };
};

class Yolov5 :public Yolo {
public:
	vector<Detection> Detect(Mat& SrcImg, Net& net);
private:
	float confidenceThreshold{ 0.25 };
	float nmsIoUThreshold{ 0.45 };
};

class Yolov7 :public Yolo {
public:
	vector<Detection> Detect(Mat& SrcImg, Net& net);
private:

	float confidenceThreshold{ 0.25 };
	float nmsIoUThreshold{ 0.45 };

	const int strideSize = 3;   //stride size
	const float netStride[4] = { 8.0, 16.0, 32.0, 64.0 };
	const float netAnchors[3][6] = { {12, 16, 19, 36, 40, 28},{36, 75, 76, 55, 72, 146},{142, 110, 192, 243, 459, 401} }; //yolov7-P5 anchors
};

class Yolov8 :public Yolo {
public:
	vector<Detection> Detect(Mat& SrcImg, Net& net);
private:
	float confidenceThreshold{ 0.25 };
	float nmsIoUThreshold{ 0.70 };
};