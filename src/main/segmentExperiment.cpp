#include <string>
#include <fstream>
#include <opencv2/opencv.hpp>
#include <func.h>
#include <config.h>
#include <opencv2/core/types_c.h>

swpr::Color getPlateColor(Mat input)
{
	cv::Mat whitePlate = cv::imread("./color/white.jpg");
	cv::Mat bluePlate = cv::imread("./color/blue.jpg");
	cv::Mat yellowPlate = cv::imread("./color/yellow.jpg");
	
	cv::Mat yellowBgrHistogram = swpr::bgrHistogram(yellowPlate);
	cv::Mat blueBgrHistogram = swpr::bgrHistogram(bluePlate);
	cv::Mat whiteBgrHistogram = swpr::bgrHistogram(whitePlate);

	Mat inputBgrHistogram = swpr::bgrHistogram(input);
	std::map<double, swpr::Color> distance;
	distance.insert(std::pair<double,swpr::Color>(
				cv::compareHist(inputBgrHistogram,whiteBgrHistogram,CV_COMP_CHISQR),swpr::WHITE));	
	distance.insert(std::pair<double,swpr::Color>(
				cv::compareHist(inputBgrHistogram,blueBgrHistogram,CV_COMP_CHISQR),swpr::BLUE));
	distance.insert(std::pair<double,swpr::Color>(
				cv::compareHist(inputBgrHistogram,yellowBgrHistogram,CV_COMP_CHISQR),swpr::WHITE));
	std::map<double,swpr::Color>::iterator i;
	double nearestDistance = (distance.begin())->first;
	swpr::Color plateColor = (distance.begin())->second;
	for(i = distance.begin();i != distance.end();++i){ 
		if(i->first < nearestDistance){ 
			nearestDistance = i->first;
			plateColor = i->second; 
		}
	}

	return plateColor;
}

swpr::Color getPlateColor2(const cv::Mat & plateImage)
{
	double pixSum = 0;
	int counterMoreThanAveragePix = 0;
	int counterLessThanAveragePix = 0;
	int nr = plateImage.rows;
	int nl = plateImage.cols*plateImage.channels();	
	std::cout << "rows : " << plateImage.rows << " cols :" << plateImage.cols << " channels :" << plateImage.channels() << std::endl;
	for(int i = 0; i < nr; ++i){
		const uchar* data = plateImage.ptr<uchar>(i);
		for(int j = 0; j < nl; ++j){
			pixSum += data[j]; 
		}
	}
	std::cout << "pixSum:" << pixSum << std::endl;
	double pixAverage = pixSum / (plateImage.rows * plateImage.cols * plateImage.channels());
	std::cout << "pixAverage:" << pixAverage << std::endl;

	for(int i = 0; i < nr; ++i){
		const uchar* data = plateImage.ptr<uchar>(i);
		for(int j = 0; j < nl; ++j){
			if(data[j] > pixAverage){
				counterMoreThanAveragePix++;
			}
			else{
				counterLessThanAveragePix++;
			}

		}
	}
	if(counterLessThanAveragePix < counterMoreThanAveragePix){	
		return swpr::YELLOW;
	}
	else{
		return swpr::BLUE;
	}
}

int main(int argc, char ** argv)
{
	if(argc < 2){
		std::cout << "You don't have input parameters!" << std::endl;
	}
	std::string fileListName = argv[1];
	std::fstream fileReader(fileListName);
	std::string fileName;
	while(fileReader >> fileName){
		std::cout << fileName << std::endl; 
		cv::Mat plateImage = cv::imread(fileName);
		cv::Mat greyImage;
		cv::cvtColor(plateImage, greyImage, cv::COLOR_BGR2GRAY);
		cv::Mat equalizeHistImage;
		cv::equalizeHist(greyImage, equalizeHistImage);
		//greyImage = equalizeHistImage;
		swpr::Color plateType = getPlateColor2(greyImage);
		cv::Mat thresholdImage = greyImage.clone();
		swpr::spatial_ostu(thresholdImage, 1, 1, plateType);
		cv::imwrite(fileName,greyImage);
	}	
	return 0;
}
