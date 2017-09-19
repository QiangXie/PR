#include "detection.hpp"     
#include "timer.hpp"


#define IS_SINGLE 0
#define RECTANGLE 1

int main(int argc, char ** argv)
{                              
	if(argc < 2){
		std::cout << "Please input list file." << std::endl;
		return -1;
	}
	std::string model_file = "../models/deploy.prototxt";
	std::string weights_file = "../models/ssd300x300_v1.caffemodel";
	std::ifstream infile(argv[1]);
	std::string img_name;
	
	
	clock_t start,end;
	int GPUID=0;               
	Timer_ timer;
	Caffe::SetDevice(GPUID);   
	Caffe::set_mode(Caffe::GPU);
	Detector det = Detector(model_file, weights_file); 
	int counter = 0;
	if(IS_SINGLE){
		vector<vector<int> > ans;  
		timer.tic();
		while(infile >> img_name){
			cv::Mat im = cv::imread(img_name);
			ans = det.Detect(im);      
			if(ans.size()){
				std::cout << "In " << img_name << " detected car at:" << std::endl;
				for(int i = 0;i < ans.size();++i){  
					for(int j = 0;j < ans[i].size();j++){
						std::cout << ans[i][j] << " ";       
					}
					std::cout << std::endl;
					if(RECTANGLE){
						cv::rectangle(im,cv::Point(ans[i][0],ans[i][1]),
								cv::Point(ans[i][2] + ans[i][0],ans[i][3] + ans[i][1]),
								cv::Scalar(0,0,255),2,1,0);
					}
				}
				if(RECTANGLE){
					cv::imwrite(img_name,im);
				}
			}
			counter++;
		}
		timer.toc();
	}
	else{
		vector<vector<vector<int> > > ans;
		timer.tic();
		std::vector<cv::Mat> imgs;
		std::vector<std::string> fileNames;
		while(infile >> img_name){
			cv::Mat im = cv::imread(img_name);
			imgs.push_back(im);
			fileNames.push_back(img_name);
			if(imgs.size() == BATCH_SIZE){
				ans = det.DetectBatch(imgs);      
				for(int i = 0;i < ans.size();++i){
					if(ans[i].size()){
						std::cout << "In " << fileNames[i] << " detected car at:" << std::endl;
						for(int j = 0;j < ans[i].size();++j){  
							for(int h = 0;h < ans[i][j].size();h++){
								std::cout << ans[i][j][h] << " ";       
							}
							std::cout << std::endl;
							if(RECTANGLE){
								cv::rectangle(imgs[i],cv::Point(ans[i][j][0],ans[i][j][1]),
										cv::Point(ans[i][j][2] + ans[i][j][0],ans[i][j][3] + ans[i][j][1]),
										cv::Scalar(0,0,255),2,1,0);
							}
						}
						if(RECTANGLE){
							cv::imwrite(fileNames[i],imgs[i]);
						}
					}
				}
				fileNames.clear();
				imgs.clear();
			}
			counter++;
		}
		timer.toc();
	}
	std::cout << "Detection " << counter << " images took " 
		<< timer.getTotalTime()/1000  << "s." << std::endl;
	std::cout << counter / (timer.getTotalTime()/1000) << " fps" << std::endl;
	return 0;
}

