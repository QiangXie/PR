#include "segment.hpp"
#include "recognition.hpp"
#include <vector>
#include <string>
#include <opencv2/opencv.hpp>
#include <codecvt>
#include <string>
#include "config.hpp"

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

using namespace swpr;


int main(int argc, char ** argv){

	string model_file = "../data/recognition_model/swnet.prototxt";
	string trained_file = "../data/recognition_model/swnet.caffemodel";
	string mean_file = "../data/recognition_model/mean.binaryproto";
	string lable_file = "../data/recognition_model/label.txt";
	Classifier classifier(model_file, trained_file, mean_file, lable_file);

	if(argc < 2){
		std::cout << "You don't have input parameters!" << std::endl;
		return -1;
	}
	std::string file = argv[1];
	std::wstring file_wstr = s2ws(file); 
	cv::Mat plate_img = imread(file);		
	std::vector<Mat> matChars;
	CharsSegment * segmenter = new CharsSegment(MASK_JPG_PATH);	
	int result = segmenter->charsSegment(plate_img, matChars);
	std::wstring file_name = L"";	
	std::vector<Prediction> ans;
	if(matChars.size() == 7){
		for(int i = 0; i != matChars.size();++i){
			file_name = L"";	
			file_name += file_wstr[i];
			string file_name_ = ws2s(file_name) + ".jpg";
			cv::imwrite(file_name_,matChars[i]);
			cv::Mat img = cv::imread(file_name_, -1);
			std::vector<Prediction> predictions = classifier.Classify(img);
			ans.push_back(predictions[0]);
		}
		std::cout << "Recognition result :" << std::endl;
		Prediction p;
		for(size_t i = 0; i < ans.size(); ++i){
			p = ans[i];
			std::cout << p.first;
		}
		std::cout << std::endl;

	}
	else{
		std::cout << "Segment fault!!!" << std::endl;
	}
	return 0;
}
