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

	string model_path1 = "models/yolov5s.onnx";
	string model_path2 = "models/yolov7.onnx";
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




	Yolov5 yolov5; Net net1;
	Mat img1 = img.clone();
	bool isOK = yolov5.readModel(net1, model_path1, USE_CUDA);
	if (isOK) {
		cout << "read net ok!" << endl;
	}
	else {
		cout << "read onnx model failed!";
		return -1;
	}
	vector<Detection> result1 = yolov5.Detect(img1, net1);
	yolov5.drawPred(img1, result1, color);
	Mat dst = img1({ 0, 0, img.cols, img.rows });
	imwrite("results/yolov5.jpg", dst);



	Yolov7 yolov7; Net net2; 
	Mat img2 = img.clone();
	isOK = yolov7.readModel(net2, model_path2, USE_CUDA);
	if (isOK) {
		cout << "read net ok!" << endl;
	}
	else {
		cout << "read onnx model failed!";
		return -1;
	}
	vector<Detection> result2 = yolov7.Detect(img2, net2);
	yolov7.drawPred(img2, result2, color);
	dst = img2({ 0, 0, img.cols, img.rows });
	imwrite("results/yolov7.jpg", dst);


	Yolov8 yolov8; Net net3;
	Mat img3 = img.clone();
	isOK = yolov8.readModel(net3, model_path3, USE_CUDA);
	if (isOK) {
		cout << "read net ok!" << endl;
	}
	else {
		cout << "read onnx model failed!";
		return -1;
	}
	vector<Detection> result3 = yolov8.Detect(img3, net3);
	yolov8.drawPred(img3, result3, color);
	dst = img3({ 0, 0, img.cols, img.rows });
	imwrite("results/yolov8.jpg", dst);


	return 0;
}