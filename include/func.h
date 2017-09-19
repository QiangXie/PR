#ifndef FUNC_H_
#define FUNC_H_
#include <vector>
#include <opencv2/opencv.hpp>
#include "character.h"
#include "config.h"

namespace swpr{
	const int HISTSIZE = 8;
	// non-maximum suppression
	void NMStoCharacter(std::vector<CCharacter> &inVec, double overlap);
	cv::Mat preprocessChar(Mat in, int char_size);
	//ostu region
	void spatial_ostu(InputArray  _src, int grid_x, int grid_y, Color type = LIGHT);
	bool clearLiuDing(Mat &img);
	float computeIOU(const cv::Rect& rect1, const cv::Rect& rect2);
	cv::Mat bgrHistogram(const cv::Mat& src);
}

#endif
