#include "segment.hpp"
#include <vector>
#include <string>
#include <opencv2/opencv.hpp>
#include <codecvt>
#include <string>
#include <fstream>
#include <unistd.h>
#include <sys/types.h>  
#include <sys/stat.h> 

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

int counter = 0;

int main(int argc, char ** argv){
	if(argc < 2){
		std::cout << "You don't have input parameters!" << std::endl;
		return -1;
	}
	std::string fileList = argv[1];
	std::fstream fileReader(fileList);
	std::string file;
	int counterSuccess = 0;
	int counterFault = 0;
	while(fileReader >> file){
		cv::Mat plate_img = imread(file);		
		std::vector<cv::Mat> matChars;
		CharsSegment * segmenter = new CharsSegment();	
		int result = segmenter->charsSegment(plate_img, matChars);
		if(matChars.size() == 7){
			std::wstring fileNameGb = s2ws(file);
			std::wstring fileJustNameGb;
			for(int i_ = fileNameGb.size() - 11; i_ < fileNameGb.size() - 4; ++i_){
				fileJustNameGb = fileJustNameGb + fileNameGb[i_];	
			}
			std::wstring singleCharFolderGb;
			std::string singleCharFolderUtf;
			std::string singleCharFolderUtfAbs;
			std::string singleCharImageName;
			for(int j_ = 0; j_ < fileJustNameGb.size(); j_++){
				singleCharFolderGb += fileJustNameGb[j_];
				singleCharFolderUtf = ws2s(singleCharFolderGb);

				singleCharFolderUtfAbs = "./singleChars/" + singleCharFolderUtf;
				//if dir can read
				if(access(singleCharFolderUtfAbs.c_str(),4) == -1){
					int mkdirFlag = mkdir(singleCharFolderUtfAbs.c_str(), 0777);
					if(mkdirFlag == -1){
						std::cout << "Make dir fault!" << std::endl;	
					}
				}
				singleCharImageName = std::to_string(counter);
				singleCharImageName = singleCharImageName + ".jpg";
				singleCharImageName = singleCharFolderUtfAbs + "/" + singleCharImageName;
				std::cout << "Save " << singleCharImageName << std::endl;
				cv::imwrite(singleCharImageName, matChars[j_]);
				counter++;

				singleCharFolderGb.clear();
				singleCharFolderUtf.clear();
				singleCharFolderUtfAbs.clear();
				singleCharImageName.clear();
			}
			counterSuccess++;
		}
		else{
			std::wstring fileNameGb = s2ws(file);
			std::wstring fileJustNameGb;
			for(int i_ = fileNameGb.size() - 11; i_ < fileNameGb.size(); ++i_){
				fileJustNameGb = fileJustNameGb + fileNameGb[i_];	
			}
			std::string fileJustNameUtf = ws2s(fileJustNameGb);
			std::string segDir = "./segFault/" + fileJustNameUtf;
			std::cout << "Save seg fault: " << segDir << std::endl;
			if(!plate_img.empty()){
				cv::imwrite(segDir,plate_img);
			}
			counterFault++;
		}
	}
	std::cout << "Segment success num: " << counterSuccess << std::endl;
	std::cout << "Segment fault num: " << counterFault << std::endl;
	return 0;
}
