#include "yolo.h"
#include <iostream>
#include<opencv2//opencv.hpp>
#include<math.h>

#define USE_CUDA false //use opencv-cuda

using namespace std;
using namespace cv;
using namespace dnn;


int main()
{
	string img_path = "images/bus.jpg";
	string model_path1 = "models/yolov7.onnx";
	string model_path2 = "models/yolov5s.onnx";
	string model_path3 = "models/yolov8n.onnx";
	Mat img = imread(img_path);


	//生成随机颜色
	vector<Scalar> color;
	srand(time(0));
	for (int i = 0; i < 80; i++) {
		int b = rand() % 256;
		int g = rand() % 256;
		int r = rand() % 256;
		color.push_back(Scalar(b, g, r));
	}

	Yolov8 yolov8; Net net3;
	Mat img1 = img.clone();
	bool isOK = yolov8.readModel(net3, model_path3, USE_CUDA);
	if (isOK) {
		cout << "read net ok!" << endl;
	}
	else {
		cout << "read onnx model failed!";
		return -1;
	}
	vector<Detection> result3 = yolov8.Detect(img1, net3);

	if (isOK) {
		yolov8.drawPred(img1, result3, color);
		Mat dst = img1({ 0, 0, img.cols, img.rows });
		imshow("yolov8", dst);
	}

	Yolov7 yolov7; Net net1; vector<Detection> result1;
	isOK = yolov7.readModel(net1, model_path1, USE_CUDA);
	if (isOK) {
		cout << "read net ok!" << endl;
	}
	else {
		cout << "read onnx model failed!";
		return -1;
	}
	result1 = yolov7.Detect(img, net1);
	yolov7.drawPred(img, result1, color);
	imshow("yolov7", img);
	


	//Yolov5 yolov5; Net net2; vector<Detection> result2; Mat img2 = img.clone();
	//isOK = yolov5.readModel(net2, model_path2, USE_CUDA);
	//if (isOK) {
	//	cout << "read net ok!" << endl;
	//}
	//else {
	//	cout << "read onnx model failed!";
	//	return -1;
	//}

	//result2 = yolov5.Detect(img2, net2);
	//yolov5.drawPred(img2, result2, color);
	//Mat dst = img2({ 0, 0, img.cols, img.rows });
	//imshow("yolov5", dst);

	waitKey(0);
	return 0;
}