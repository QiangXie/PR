#include "recognition.hpp"
#include <string>
using namespace std;
using namespace swpr;

int main(int argc,char ** argv){
	string model_file = "../data/recognition_model/swnet.prototxt";
	string trained_file = "../data/recognition_model/swnet.caffemodel";
	string mean_file = "../data/recognition_model/mean.binaryproto";
	string lable_file = "../data/recognition_model/label.txt";
	Classifier classifier(model_file, trained_file, mean_file, lable_file);

	if(argc < 2){
		std::cerr << "Usage: " << argv[0]
			  << "img.jpg" << std::endl;
		return 1;
	}
	::google::InitGoogleLogging(argv[0]);
	string file = argv[1];
	std::cout << "-------------Prediction for " << file << " ----------"
		  << std::endl;
	cv::Mat img = cv::imread(file,-1);
	CHECK(!img.empty()) << "Unable to decode image " << file;
	std::vector<Prediction> predictions = classifier.Classify(img);
	for(size_t i = 0;i < predictions.size(); ++i){
		Prediction p = predictions[i];
		std::cout << std::fixed << std::setprecision(4) << p.second << " - \""
			  << p.first << "\"" << std::endl;
	}
}
