#include "pr.hpp"
#include <iostream>

using namespace std;
using namespace swpr;

const string list_path = "./list.txt";

int main(){
	int num_of_vehicle_detected = 0;
	int num_of_rec_right = 0;
	std::string vehicle_detector_model_file = "../models/detector_model/deploy_2class.prototxt";
	std::string vehicle_detector_weights_file = "../models/detector_model/ssd300x300_plate_2class_v1.caffemodel";
	Detector vehicle_detector = Detector(detector_model_file, detector_weights_file);

	ifstream list(list_path);
	string path;
	plateRecognizer * pr = new plateRecognizer;
	while(list >> path){
		cv::Mat img_cv = cv::imread(path);
		vehicles = vehicle_detector.Detect(img_cv);
		if(vehicles.size()){
			vector<string> rec_results;
			rec_results.clear();
			num_of_vehicle_detected++;
			for(int i = 0; i < vehicles.size(); i++){
				int plate_x = plates[i][0];
				int plate_y = plates[i][1];
				int plate_w = plates[i][2];
				int plate_h = plates[i][3];
				if(plate_x > im_cv.cols || plate_y > im_cv.rows || plate_w <= 0 || plate_h <= 0){
					LOG(INFO) << "Detect result error!";
				}
				else{
					if((plate_x + plate_w) > im_cv.cols ){
						plates[0][2] = im_cv.cols - plate_x;
					}
					if((plate_y + plate_h) > im_cv.rows){
						plates[0][3] = im_cv.rows - plate_y;
					}
					cv::Rect vehicle_roi(plates[0][0],plates[0][1],plates[0][2],plates[0][3]);
					cv::Mat vehicle_im_cv = im_cv(vehicle_roi);
					string rec_result = "";
					int flag = pr->plateRecognize(vehicle_im_cv,rec_result);
					if(flag == 0){
						rec_results.push_back(rec_result);
					}
				}
			}
			for(int i = 0; i < rec_results.size(); ++i){
					
			}

		}
	}
	delete pr;

	return 0;
}

