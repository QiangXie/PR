#include "pr.hpp"
#include <iostream>

using namespace std;
using namespace swpr;

const string list_path = "./its_list.txt";

int main(){
	int num_of_vehicle_detected = 0;
	int num_of_rec_right = 0;
	std::string vehicle_detector_model_file = "../models/detector_model/vehicle_models/deploy.prototxt";
	std::string vehicle_detector_weights_file = "../models/detector_model/vehicle_models/ssd300x300_v1.caffemodel";
	Detector vehicle_detector = Detector(detector_model_file, detector_weights_file, 7, 0.3);

	ifstream list(list_path);
	string path;
	plateRecognizer * pr = new plateRecognizer;
	while(list >> path){
		string gt = "";
		int index = path.size() - 1;
		while(path[index] != '.'){
			index--;	
		}
		for(int index_ = index - 1; path[index_] != '/'; index_--){
			gt = path[index_] + gt;	
		}
		LOG(INFO) << "Image path: " << path;
		LOG(INFO) << "Ground truth: " << gt;
		cv::Mat img_cv = cv::imread(path);
		
		vector<vector<int> > vehicles = vehicle_detector.Detect(img_cv);
		if(vehicles.size()){
			vector<string> rec_results;
			rec_results.clear();
			num_of_vehicle_detected++;
			for(int i = 0; i < vehicles.size(); i++){
				int plate_x = vehicles[i][0];
				int plate_y = vehicles[i][1];
				int plate_w = vehicles[i][2];
				int plate_h = vehicles[i][3];
				if(plate_x > img_cv.cols || plate_y > img_cv.rows || plate_w <= 0 || plate_h <= 0){
					LOG(INFO) << "Detect result error!";
				}
				else{
					if((plate_x + plate_w) > img_cv.cols ){
						vehicles[i][2] = img_cv.cols - plate_x;
					}
					if((plate_y + plate_h) > img_cv.rows){
						vehicles[i][3] = img_cv.rows - plate_y;
					}
					cv::Rect vehicle_roi(vehicles[i][0],vehicles[i][1],vehicles[i][2],vehicles[i][3]);
					cv::Mat vehicle_img_cv = img_cv(vehicle_roi);
					string rec_result = "";
					int flag = pr->plateRecognize(vehicle_img_cv,rec_result);
					if(flag == 0){
						rec_results.push_back(rec_result);
					}
				}
			}
			for(int i = 0; i < rec_results.size(); ++i){
				if(gt == rec_results[i]){
					num_of_rec_right++;
					break;
				}	
			}

		}
	}
	delete pr;
	LOG(INFO) << "Right: " << num_of_rec_right;
	LOG(INFO) << "All:" << num_of_vehicle_detected;
	LOG(INFO) << "accuracy:" << static_cast<float>(num_of_rec_right) / static_cast<float>(num_of_vehicle_detected);


	return 0;
}

