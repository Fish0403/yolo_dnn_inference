#include"yolo.h"

bool Yolo::readModel(Net& net, string& netPath, bool isCuda = false) {
	try {
		net = readNetFromONNX(netPath);
	}
	catch (const std::exception&) {
		return false;
	}
	if (isCuda) {
		net.setPreferableBackend(cv::dnn::DNN_BACKEND_CUDA);
		net.setPreferableTarget(cv::dnn::DNN_TARGET_CUDA);
	}
	else {
		net.setPreferableBackend(cv::dnn::DNN_BACKEND_DEFAULT);
		net.setPreferableTarget(cv::dnn::DNN_TARGET_CPU);
	}
	return true;
}

void Yolo::drawPred(Mat& img, vector<Detection> result, vector<Scalar> colors) {

	for (int i = 0; i < result.size(); ++i)
	{
		Detection detection = result[i];

		cv::Rect box = detection.box;
		cv::Scalar color = colors[detection.class_id];

		// Detection box
		cv::rectangle(img, box, color, 2);

		// Detection box text
		std::string classString = classes[detection.class_id] + ' ' + std::to_string(detection.confidence).substr(0, 4);
		cv::Size textSize = cv::getTextSize(classString, cv::FONT_HERSHEY_DUPLEX, 1, 2, 0);
		cv::Rect textBox(box.x, box.y - 40, textSize.width + 10, textSize.height + 20);

		cv::rectangle(img, textBox, color, cv::FILLED);
		cv::putText(img, classString, cv::Point(box.x + 5, box.y - 10), cv::FONT_HERSHEY_DUPLEX, 1, cv::Scalar(0, 0, 0), 2, 0);
	}
}

vector<Detection> Yolov5::Detect(Mat& img, Net& net) {

	img = formatToSquare(img);
	cv::Mat blob;
	cv::dnn::blobFromImage(img, blob, 1.0 / 255.0, Size(netWidth, netHeight), cv::Scalar(), true, false);
	net.setInput(blob);

	std::vector<cv::Mat> outputs;
	net.forward(outputs, net.getUnconnectedOutLayersNames());



	float* pdata = (float*)outputs[0].data;
	float x_factor = (float)img.cols / netWidth;
	float y_factor = (float)img.rows / netHeight;

	std::vector<int> class_ids;
	std::vector<float> confidences;
	std::vector<cv::Rect> boxes;

	// yolov5 has an output of shape (batchSize, 25200, 85) (Num classes + box[x,y,w,h] + confidence[c])
	int rows = outputs[0].size[1];
	int dimensions = outputs[0].size[2];

	for (int i = 0; i < rows; ++i)
	{
		float confidence = pdata[4];
		if (confidence >= modelConfidenceThreshold)
		{

			cv::Mat scores(1, classes.size(), CV_32FC1, pdata + 5);
			cv::Point class_id;
			double max_class_score;

			minMaxLoc(scores, 0, &max_class_score, 0, &class_id);

			if (max_class_score > modelScoreThreshold)
			{
				confidences.push_back(confidence);
				class_ids.push_back(class_id.x);

				float x = pdata[0];
				float y = pdata[1];
				float w = pdata[2];
				float h = pdata[3];

				int left = int((x - 0.5 * w) * x_factor);
				int top = int((y - 0.5 * h) * y_factor);

				int width = int(w * x_factor);
				int height = int(h * y_factor);

				boxes.push_back(cv::Rect(left, top, width, height));
			}
		}
		pdata += dimensions;
	}

	vector<int> nms_result;
	NMSBoxes(boxes, confidences, confidenceThreshold, nmsIoUThreshold, nms_result);
	vector<Detection> detections{};
	for (unsigned long i = 0; i < nms_result.size(); ++i) {
		int idx = nms_result[i];
		Detection result;
		result.class_id = class_ids[idx];
		result.confidence = confidences[idx];
		result.box = boxes[idx];
		detections.push_back(result);
	}
	return detections;
}

vector<Detection> Yolov7::Detect(Mat& SrcImg, Net& net) {
	Mat blob;
	int col = SrcImg.cols;
	int row = SrcImg.rows;
	int maxLen = MAX(col, row);
	Mat netInputImg = SrcImg.clone();
	if (maxLen > 1.2 * col || maxLen > 1.2 * row) {
		Mat resizeImg = Mat::zeros(maxLen, maxLen, CV_8UC3);
		SrcImg.copyTo(resizeImg(Rect(0, 0, col, row)));
		netInputImg = resizeImg;
	}
	vector<Ptr<Layer> > layer;
	vector<string> layer_names;
	layer_names = net.getLayerNames();
	blobFromImage(netInputImg, blob, 1 / 255.0, cv::Size(netWidth, netHeight), cv::Scalar(0, 0, 0), true, false);
	net.setInput(blob);
	std::vector<cv::Mat> netOutput;
	net.forward(netOutput, net.getUnconnectedOutLayersNames());
#if CV_VERSION_MAJOR==4 && CV_VERSION_MINOR>6
	std::sort(netOutput.begin(), netOutput.end(), [](Mat& A, Mat& B) {return A.size[2] > B.size[2]; });//opencv 4.6 三个output顺序为40 20 80，之前版本的顺序为80 40 20 
#endif
	std::vector<int> class_ids;//结果id数组
	std::vector<float> confidences;//结果每个id对应置信度数组
	std::vector<cv::Rect> boxes;//每个id矩形框
	float ratio_h = (float)netInputImg.rows / netHeight;
	float ratio_w = (float)netInputImg.cols / netWidth;
	int net_width = classes.size() + 5;  //输出的网络宽度是类别数+5
	for (int stride = 0; stride < strideSize; stride++) {    //stride
		float* pdata = (float*)netOutput[stride].data;
		int grid_x = (int)(netWidth / netStride[stride]);
		int grid_y = (int)(netHeight / netStride[stride]);
		// cv::Mat tmp(grid_x * grid_y * 3, classes.size() + 5, CV_32FC1, pdata);
		for (int anchor = 0; anchor < 3; anchor++) {	//anchors
			const float anchor_w = netAnchors[stride][anchor * 2];
			const float anchor_h = netAnchors[stride][anchor * 2 + 1];
			for (int i = 0; i < grid_y; i++) {
				for (int j = 0; j < grid_x; j++) {
					float confidence = sigmoid_x(pdata[4]); ;//获取每一行的box框中含有物体的概率
					cv::Mat scores(1, classes.size(), CV_32FC1, pdata + 5);
					Point classIdPoint;
					double max_class_socre;
					minMaxLoc(scores, 0, &max_class_socre, 0, &classIdPoint);
					max_class_socre = sigmoid_x(max_class_socre);
					if (max_class_socre*confidence >= confidenceThreshold) {
					float x = (sigmoid_x(pdata[0]) * 2.f - 0.5f + j) * netStride[stride];  //x
					float y = (sigmoid_x(pdata[1]) * 2.f - 0.5f + i) * netStride[stride];   //y
					float w = powf(sigmoid_x(pdata[2]) * 2.f, 2.f) * anchor_w;   //w
					float h = powf(sigmoid_x(pdata[3]) * 2.f, 2.f) * anchor_h;  //h
					int left = (int)(x - 0.5 * w) * ratio_w + 0.5;
					int top = (int)(y - 0.5 * h) * ratio_h + 0.5;
					class_ids.push_back(classIdPoint.x);
					confidences.push_back(max_class_socre * confidence);
					boxes.push_back(Rect(left, top, int(w * ratio_w), int(h * ratio_h)));
					}
					pdata += net_width;//下一行
				}
			}
		}
	}

	//执行非最大抑制以消除具有较低置信度的冗余重叠框（NMS）
	vector<int> nms_result;
	NMSBoxes(boxes, confidences, confidenceThreshold, nmsIoUThreshold, nms_result);
	vector<Detection> detections{};
	for (unsigned long i = 0; i < nms_result.size(); ++i) {
		int idx = nms_result[i];
		Detection result;
		result.class_id = class_ids[idx];
		result.confidence = confidences[idx];
		result.box = boxes[idx];
		detections.push_back(result);
	}
	return detections;
}

vector<Detection> Yolov8::Detect(Mat& modelInput, Net& net) {
	modelInput = formatToSquare(modelInput);
	cv::Mat blob;
	cv::dnn::blobFromImage(modelInput, blob, 1.0 / 255.0, Size(netWidth, netHeight), cv::Scalar(), true, false);
	net.setInput(blob);

	std::vector<cv::Mat> outputs;
	net.forward(outputs, net.getUnconnectedOutLayersNames());


	// yolov8 has an output of shape (batchSize, 84,  8400) (Num classes + box[x,y,w,h])
	int rows = outputs[0].size[2];
	int dimensions = outputs[0].size[1];

	outputs[0] = outputs[0].reshape(1, dimensions);
	cv::transpose(outputs[0], outputs[0]);
	
	float* data = (float*)outputs[0].data;
	// Mat detect_output(8400, 84, CV_32FC1, data);// 8400 = 80*80+40*40+20*20
	float x_factor = (float)modelInput.cols / netWidth;
	float y_factor = (float)modelInput.rows / netHeight;

	std::vector<int> class_ids;
	std::vector<float> confidences;
	std::vector<cv::Rect> boxes;

	for (int i = 0; i < rows; ++i)
	{
		cv::Mat scores(1, classes.size(), CV_32FC1, data + 4);
		cv::Point class_id;
		double maxClassScore;

		minMaxLoc(scores, 0, &maxClassScore, 0, &class_id);
		if (maxClassScore > modelConfidenceThreshold)
		{
			confidences.push_back(maxClassScore);
			class_ids.push_back(class_id.x);
			
			float x = data[0];
			float y = data[1];
			float w = data[2];
			float h = data[3];

			int left = int((x - 0.5 * w) * x_factor);
			int top = int((y - 0.5 * h) * y_factor);

			int width = int(w * x_factor);
			int height = int(h * y_factor);

			boxes.push_back(cv::Rect(left, top, width, height));
		}
		data += dimensions;
	}

	//执行非最大抑制以消除具有较低置信度的冗余重叠框（NMS）
	vector<int> nms_result;
	NMSBoxes(boxes, confidences, confidenceThreshold, nmsIoUThreshold, nms_result);
	vector<Detection> detections{};
	for (unsigned long i = 0; i < nms_result.size(); ++i){
		int idx = nms_result[i];
		Detection result;
		result.class_id = class_ids[idx];
		result.confidence = confidences[idx];
		result.box = boxes[idx];
		detections.push_back(result);
	}
	return detections;
}
