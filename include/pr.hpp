#ifndef __PR_HPP__
#define __PR_HPP__

#include "detection.hpp"
#include "recognition.hpp"
#include <opencv2/opencv.hpp>
#include <codecvt>
#include "segment.hpp"
#include <gflags/gflags.h>
#include <glog/logging.h>

using namespace cv;

namespace swpr{
	class plateRecognizer{
		public:
			explicit plateRecognizer();
			~plateRecognizer();
			int plateRecognize(const Mat & vehicleImage, string & result);
		private:
			Detector * plate_detector;
			Classifier * classifier;
			CharsSegment * segmenter;
	};
}

#endif
