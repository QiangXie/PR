#ifndef _CONFIG_HPP_
#define _CONFIG_HPP_
#include <string>

namespace swpr{
	enum Color { DEEP, LIGHT};
	extern const std::string MASK_JPG_PATH;
	extern const int GPU_ID;
	extern const bool SAVE_SEGMENT_FLAG;
	//extern const int  DETEC_CLS_NUM;
	//extern const float CONF_THRESH;
	extern const int BATCH_SIZE;
	extern const int kChineseSize;
	extern std::string detector_model_file;
	extern std::string detector_weights_file;
	extern std::string model_file;
	extern std::string trained_file;
	extern std::string mean_file;
	extern std::string lable_file;
}

#endif
