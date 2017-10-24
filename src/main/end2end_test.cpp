#include "detection.hpp"
#include "recognition.hpp"
#include <opencv2/opencv.hpp>
#include <codecvt>
#include "segment.hpp"
#include <gflags/gflags.h>
#include <glog/logging.h>
#include "config.hpp"

using namespace swpr;
#define RECTANGLE 0
#define THRESHOLD_DETECT_IOU 0.8

DEFINE_int32(gpu, 11, 
		"The GPU ID to be used");
DEFINE_string(img_list, "./list.txt",
		"List of images to be test");
DEFINE_string(name2id, "/home/swli/Data/end2endPlateTest/name2id.txt",
		"Name to ID list");


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


long num_of_imgs = 0;
long num_of_segment_fault = 0;
long num_of_single_char_rec_fault = 0;
long num_of_plate_rec_fault = 0;
long num_of_6b_en_char_rec_fault = 0;
long num_of_detect_fault = 0;

int main(int argc, char ** argv)
{
	gflags::ParseCommandLineFlags(&argc, &argv, true);
	std::ifstream infile(FLAGS_img_list.c_str());
	std::string vehicle_image;
	std::ifstream name2id_ifs_obj(FLAGS_name2id);
	std::map<std::string, std::string> map_name2id;
	std::string name,id;
	while(name2id_ifs_obj >> name){
		name2id_ifs_obj >> id;
		map_name2id[name] = id;
		LOG(INFO) << "Image name: " << name << "image id: " << id;
	}

	//detector model path	
	std::string detector_model_file = "../models/detector_model/deploy_2class.prototxt";
	std::string detector_weights_file = "../models/detector_model/ssd300x300_plate_2class_v1.caffemodel";
	//classifier model path
	string model_file = "../models/recognition_model/swnet.prototxt";
	string trained_file = "../models/recognition_model/swnet_v2.0.caffemodel";
	string mean_file = "../models/recognition_model/mean.binaryproto";
	string lable_file = "../models/recognition_model/label.txt";

	Caffe::SetDevice(FLAGS_gpu);
	Caffe::set_mode(Caffe::GPU);
	//init detector
	LOG(INFO) << "Start init detector...";
	LOG(INFO) << "Start init detector...";
	Detector plate_detector = Detector(detector_model_file, detector_weights_file);
	LOG(INFO) << "Detector init success!";
	//init classifier
	LOG(INFO) << "Start init classifier...";
	Classifier classifier(model_file, trained_file, mean_file, lable_file);
	LOG(INFO) << "Classifier init success!";
	//init segmenter
	CharsSegment * segmenter = new CharsSegment(MASK_JPG_PATH);

	vector<vector<int> > plates;
	std::string img_name;
	while(infile >> vehicle_image){
		img_name = "";
		num_of_imgs += 1;
		cv::Mat vehicle_im_cv = cv::imread(vehicle_image);
		LOG_IF(WARNING,vehicle_image.empty()) << vehicle_image << "is empty.";
		for(int i = vehicle_image.size() - 1;vehicle_image[i] != '/';i--){
			img_name = vehicle_image[i] + img_name;
		}

		plates = plate_detector.Detect(vehicle_im_cv);
		if(plates.size() == 1){

			int plate_x = plates[0][0];
			int plate_y = plates[0][1];
			int plate_w = plates[0][2];
			int plate_h = plates[0][3];
			if(plate_x > vehicle_im_cv.cols || plate_y > vehicle_im_cv.rows || plate_w <= 0 || plate_h <= 0){
				num_of_detect_fault += 1;
				LOG(INFO) << "Detect result error!";
			}
			else{
				if((plate_x + plate_w) > vehicle_im_cv.cols ){
					plates[0][2] = vehicle_im_cv.cols - plate_x;
				}
				if((plate_y + plate_h) > vehicle_im_cv.rows){
					plates[0][3] = vehicle_im_cv.rows - plate_y;
				}
				LOG(INFO) << "In " << vehicle_image << " detected plate at:" 
					<< plates[0][0] << " " <<  plates[0][1] <<  " " << plates[0][2] <<  " " << plates[0][3];

				cv::Rect plate_roi(plates[0][0],plates[0][1],plates[0][2],plates[0][3]);
				cv::Mat plate_im_cv = vehicle_im_cv(plate_roi);
				cv::Mat plate_roi_resize;
				cv::resize(plate_im_cv, plate_roi_resize, cv::Size(120, 30),0,0, cv::INTER_LINEAR);
				std::vector<cv::Mat> matChars;
				int seg_result = segmenter->charsSegment(plate_roi_resize, matChars);  
				LOG_IF(INFO, matChars.size() != 7) << "Segment fault! The size of chars is " << matChars.size();
				if(matChars.size() != 7){
					num_of_segment_fault += 1;
				}
				else{
					std::vector<Prediction> predictions;
					for(int j = 0; j != matChars.size();++j){
						std::vector<Prediction> single_predictions = classifier.Classify(matChars[j]);
						predictions.push_back(single_predictions[0]);
					}
					Prediction p;
					bool last6b_fault_flag = false;
					bool plate_reg_flag = false;
					std::string id__ = map_name2id[img_name];
					std::wstring id__ws = s2ws(id__);
					CHECK_EQ(id__ws.size(),predictions.size()) << "The size of id not equal predictions's size.";
					std::string rec_result;
					for(size_t j = 0;j < predictions.size();++j){
						std::wstring id_single_ws = L"";
						id_single_ws += id__ws[j];
						std::string id_single = ws2s(id_single_ws);
						p = predictions[j];
						if(p.first != id_single){
							plate_reg_flag = true;
							num_of_single_char_rec_fault += 1;
							if(j >= 1 && j <= 6){
								last6b_fault_flag = true;
							}
						}
						rec_result += p.first;
					}
					LOG(INFO) << "Plate rec result: " << rec_result;
					LOG(INFO)  << "Ground truth : " << map_name2id[img_name];
					if(last6b_fault_flag){
						num_of_6b_en_char_rec_fault += 1;
					}
					if(plate_reg_flag){
						num_of_plate_rec_fault += 1;
					}

				}
				if(RECTANGLE){		
					cv::rectangle(vehicle_im_cv,
							cv::Point(plates[0][0],plates[0][1]),
							cv::Point(plates[0][0]+plates[0][2],plates[0][1]+plates[0][3]),
							cv::Scalar(0,0,255),2,1,0);
					cv::imwrite("vehicle_image_rec.jpg",vehicle_im_cv);
				}
				plates.clear();
			}
		}
		else{
			LOG(INFO)  << "Plate detect error, " << plates.size() << "plate detected.";
			plates.clear();
			num_of_detect_fault += 1;
		}
	}
	LOG(INFO) << "Test " << num_of_imgs << " images.";
	LOG(INFO) << "Detection fault num: " << num_of_detect_fault;
	LOG(INFO) << "Segment fault num: " << num_of_segment_fault;
	LOG(INFO) << "Rec fault num: " << num_of_plate_rec_fault;
	LOG(INFO) << "Detector accuracy: " << 
		static_cast<float>(num_of_imgs - num_of_detect_fault) / 
		static_cast<float>(num_of_imgs);
	LOG(INFO) << "Segment accuracy: " << 
		static_cast<float>(num_of_imgs - num_of_detect_fault - num_of_segment_fault) / 
		static_cast<float>(num_of_imgs - num_of_detect_fault);
	LOG(INFO) << "Single char rec accuracy: " << 
		static_cast<float>((num_of_imgs - num_of_detect_fault - num_of_segment_fault)*7 - num_of_single_char_rec_fault) / 
		static_cast<float>((num_of_imgs - num_of_detect_fault - num_of_segment_fault)*7);
	LOG(INFO) << "Last 6bit rec  accuracy: " << 
		static_cast<float>((num_of_imgs - num_of_detect_fault - num_of_segment_fault) - num_of_6b_en_char_rec_fault) / 
		static_cast<float>(num_of_imgs - num_of_detect_fault - num_of_segment_fault); 
	LOG(INFO)  << "Plate rec  accuracy: " << 
		static_cast<float>(num_of_imgs - num_of_detect_fault - num_of_segment_fault - num_of_plate_rec_fault) / 
		static_cast<float>(num_of_imgs - num_of_detect_fault - num_of_segment_fault); 
	LOG(INFO)  << "Whole system accuracy: " << 
		static_cast<float>(num_of_imgs - num_of_detect_fault - num_of_segment_fault - num_of_plate_rec_fault) / 
		static_cast<float>(num_of_imgs); 

	return 0;
}
