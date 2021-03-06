#include "config.hpp"
#include <string>

namespace swpr{
	const int GPU_ID = 7;
	const bool SAVE_SEGMENT_FLAG = true;
	const std::string MASK_JPG_PATH = "../models/segment_model/maks_bin_0.jpg";
	//DEEP means char color deeper than background

	const int kChineseSize = 28;
	//detector model path	
	std::string detector_model_file = "../models/detector_model/deploy_2class.prototxt";
	std::string detector_weights_file = "../models/detector_model/ssd300x300_plate_2class_v1.caffemodel";
	//classifier model path
	std::string model_file = "../models/recognition_model/swnet.prototxt";
	std::string trained_file = "../models/recognition_model/swnet_v2.0.caffemodel";
	std::string mean_file = "../models/recognition_model/mean.binaryproto";
	std::string lable_file = "../models/recognition_model/label.txt";
}

