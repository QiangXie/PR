#ifndef _DETECTION_HPP_
#define _DETECTION_HPP_
#include <caffe/caffe.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <algorithm>
#include <iomanip>
#include <iosfwd>
#include <memory>
#include <string>
#include <utility>
#include <vector>
#include "timer.hpp"

using namespace caffe;

namespace swpr{

	class Detector{
		public:
			Detector(const string & model_file,
				 const string & weights_file,
				 int class_num = 1,
				 float conf_thresh = 0.2,
				 int batch_size = 8);
			std::vector<std::vector<int> > Detect(const cv::Mat & img);
			std::vector<std::vector<std::vector<int> > > DetectBatch(const std::vector<cv::Mat> & imgs);
		private:
			void SetMean(const string& mean_file, const string& mean_value);
			void WrapInputLayerBatch(std::vector< std::vector<cv::Mat> >* input_batch);
			void WrapInputLayer(std::vector<cv::Mat>* input_channels);
			void PreprocessBatch(const vector<cv::Mat> imgs,std::vector< std::vector<cv::Mat> >* input_batch);
			void Preprocess(const cv::Mat& img,std::vector<cv::Mat>* input_channels);

			shared_ptr<Net<float> > net_;
			cv::Size input_geometry_;
			int num_channels_;
			cv::Mat mean_;
			const int detec_cls_num;
			const float conf_thresh;
			const int batch_size; 

	};

}

#endif
