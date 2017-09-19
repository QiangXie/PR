#include "chars_segment.h" 
#include "func.h"
#include <map>
#include <codecvt>
#include <string>
#include <fstream>
#include <opencv2/opencv.hpp> 

std::wstring s2ws(const std::string& str)
{
	using convert_typeX = std::codecvt_utf8<wchar_t>;
	std::wstring_convert<convert_typeX, wchar_t> converterX;

	return converterX.from_bytes(str);
}

std::string ws2s(const std::wstring& wstr)
{
	using convert_typeX = std::codecvt_utf8<wchar_t>;
	std::wstring_convert<convert_typeX, wchar_t> converterX;

	return converterX.to_bytes(wstr);
}


int main()
{
	std::string list_path = "./list.txt";
	std::string black_folder = "/media/mmr6-raid5/xieqiang/color/black/";
	std::string blue_folder = "/media/mmr6-raid5/xieqiang/color/blue/";
	std::string white_folder = "/media/mmr6-raid5/xieqiang/color/white/";
	std::string yello_folder = "/media/mmr6-raid5/xieqiang/color/yellow/";

	std::string white_path = "./white.jpg";
	std::string yellow_path = "./yellow.jpg";
	std::string black_path = "./black.jpg";
	std::string blue_path = "./blue.jpg";


	std::string img_name;
	std::ifstream infile(list_path);
	while(infile >> img_name){  
		std::wstring ws_img_name = s2ws(img_name);
		std::wstring ws_just_name;

		for(int i = ws_img_name.size() - 11;i < ws_img_name.size();++i){
			ws_just_name += ws_img_name[i];
		}	
		cv::Mat im = cv::imread(img_name);
		std::map<double,int> distance;
		cv::Mat Blue_bgrHistogram = swpr::bgrHistogram(cv::imread(blue_path));
		cv::Mat Yellow_bgrHistogram = swpr::bgrHistogram(cv::imread(yellow_path));
		cv::Mat Black_bgrHistogram = swpr::bgrHistogram(cv::imread(black_path));
		cv::Mat White_bgrHistogram = swpr::bgrHistogram(cv::imread(white_path));

		cv::Mat in_bgrHistogram = swpr::bgrHistogram(im);

		distance.insert(std::pair<double,int>(cv::compareHist(in_bgrHistogram,Blue_bgrHistogram,CV_COMP_CHISQR),0));
		distance.insert(std::pair<double,int>(cv::compareHist(in_bgrHistogram,Yellow_bgrHistogram,CV_COMP_CHISQR),1));
		distance.insert(std::pair<double,int>(cv::compareHist(in_bgrHistogram,Black_bgrHistogram,CV_COMP_CHISQR),2));
		distance.insert(std::pair<double,int>(cv::compareHist(in_bgrHistogram,White_bgrHistogram,CV_COMP_CHISQR),3));

		double nearestDistance = (distance.begin())->first;
		int color_id = (distance.begin())->second;
		for(std::map<double,int>::iterator i = distance.begin();i != distance.end();++i){
			if(i->first < nearestDistance){
				nearestDistance = i->first;
				color_id = i->second;
			}	
		}

		switch (color_id){
			case 0:
				cv::imwrite(ws2s(s2ws(blue_folder) + ws_just_name),im);
				break;
			case 1:
				cv::imwrite(ws2s(s2ws(yello_folder) + ws_just_name),im);
				break;
			case 2:
				cv::imwrite(ws2s(s2ws(black_folder) + ws_just_name),im);
				break;
			case 3:
				cv::imwrite(ws2s(s2ws(white_folder) + ws_just_name),im);
				break;
		}
	}
		
	return 0;
}

