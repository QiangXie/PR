#include "pr.hpp"
#include <iostream>

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
		if(pr->plateRecognize(vehicle_img_cv,rec_result) == 0){
			LOG(INFO) << path << " : " << rec_result;
		}
	}
	delete pr;

	return 0;
}

