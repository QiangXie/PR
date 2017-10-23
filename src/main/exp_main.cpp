#include "pr.hpp"
#include "detection.hpp"
#include <iostream>
#include <string>


using namespace std;
using namespace swpr;

const string list_path = "./list.txt";

int main(){

	ifstream list(list_path);
	string path;
	plateRecognizer * pr = new plateRecognizer;
	while(list >> path){
		cv::Mat vehicle_img_cv = cv::imread(path);
		string rec_result = "";
		int flag = pr->plateRecognize(vehicle_img_cv,rec_result);
		if(flag == 0){
			LOG(INFO) << path << " : " << rec_result;
		}
	}
	return 0;
}
