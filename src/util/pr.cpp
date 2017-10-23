#include <pr.hpp>

namespace swpr{
	plateRecognizer::plateRecognizer(){
		Caffe::SetDevice(GPU_ID);
		Caffe::set_mode(Caffe::GPU);
		this->plate_detector = new Detector(detector_model_file, detector_weights_file);
		this->classifier = new Classifier(model_file, trained_file, mean_file, lable_file); 
		this->segmenter = new CharsSegment();
		LOG(INFO) << "Plate recognizer init success!";
	}	
	plateRecognizer::~plateRecognizer(){
		delete this->segmenter;
		delete this->plate_detector;
		delete this->classifier;
	}
	int plateRecognizer::plateRecognize(const Mat & vehicleImage, string & result){
		vector<vector<int> > plates_detected;
		plates_detected = this->plate_detector->Detect(vehicleImage);
		if(plates_detected.size() != 1){
			LOG(INFO) << "Error: number of plate detected more than 1.";
			return 0x01;
		}
		else{
			int plate_x = plates_detected[0][0];
			int plate_y = plates_detected[0][1];
			int plate_w = plates_detected[0][2];
			int plate_h = plates_detected[0][3];
			if(plate_w > vehicleImage.cols || plate_y > vehicleImage.rows ||
					plate_h <= 0 || plate_w <= 0){
				LOG_IF(INFO, plate_w > vehicleImage.cols) << 
					"plate_w > vehicleImage.cols";
				LOG_IF(INFO, plate_y > vehicleImage.rows) << 
					"plate_y > vehicleImage.rows";
				LOG_IF(INFO, plate_h <= 0) << "plate_h <= 0";
				LOG_IF(INFO, plate_w <= 0) << "plate_w <= 0";
				return 0x02;
			}
			if((plate_x + plate_w) > vehicleImage.cols ){ 
				plates_detected[0][2] = vehicleImage.cols - plate_x;
			}
			if((plate_y + plate_h) > vehicleImage.rows){
				plates_detected[0][3] = vehicleImage.rows - plate_y;
			}
			cv::Rect plate_roi(plates_detected[0][0],plates_detected[0][1],
					plates_detected[0][2],plates_detected[0][3]); 
			cv::Mat plate_im_cv = vehicleImage(plate_roi);
			cv::Mat plate_roi_resize;
			cv::resize(plate_im_cv, plate_roi_resize, cv::Size(120, 30),0,0, cv::INTER_LINEAR);
			std::vector<cv::Mat> matChars;
			int seg_result = this->segmenter->charsSegment(plate_roi_resize, matChars);
			if(matChars.size() == 0){
				LOG(INFO) << "Segment result size = 0.";
				return 0x03;
			}
			else{
				std::vector<Prediction> predictions;
				for(int j = 0; j < matChars.size();++j){
					std::vector<Prediction> single_predictions = this->classifier->Classify(matChars[j]);
					predictions.push_back(single_predictions[0]);
				}

				result.clear();
				for(int j = 0; j < predictions.size(); ++j){
					result += predictions[j].first;
				}
				if(matChars.size() != 7){
					return 0x04;
				}
				else{
					return 0;
				}
			}
		}
	}
}
