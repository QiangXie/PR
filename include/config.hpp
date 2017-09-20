#ifndef _CONFIG_HPP_
#define _CONFIG_HPP_
namespace swpr{
	const bool SAVE_SEGMENT_FLAG = true;
	const int  DETEC_CLS_NUM = 1;
	const float CONF_THRESH = 0.2;
	const int BATCH_SIZE = 8;
	//DEEP means char color deeper than background
	enum Color { DEEP, LIGHT};
	
	const int kChineseSize = 28;
}

#endif
