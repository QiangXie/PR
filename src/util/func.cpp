#include <func.h>
#include <vector>

using namespace cv;

namespace swpr{
	//! non-maximum suppression
	void NMStoCharacter(std::vector<CCharacter> &inVec, double overlap) {
		std::sort(inVec.begin(), inVec.end());
		std::vector<CCharacter>::iterator it = inVec.begin();
		for (; it != inVec.end(); ++it) {
			CCharacter charSrc = *it;
			Rect rectSrc = charSrc.getCharacterPos();

			std::vector<CCharacter>::iterator itc = it + 1;
			for (; itc != inVec.end();) {
				CCharacter charComp = *itc;
				Rect rectComp = charComp.getCharacterPos();
				float iou = computeIOU(rectSrc, rectComp);
				if (iou > overlap) {
					itc = inVec.erase(itc);
				}
				else {
					++itc;
				}
			}
		}
	}

	Mat preprocessChar(Mat in, int char_size) {
		// Remap image
		int h = in.rows;
		int w = in.cols;

		int charSize = char_size;

		Mat transformMat = Mat::eye(2, 3, CV_32F);
		int m = max(w, h);
		transformMat.at<float>(0, 2) = float(m / 2 - w / 2);
		transformMat.at<float>(1, 2) = float(m / 2 - h / 2);

		Mat warpImage(m, m, in.type());
		warpAffine(in, warpImage, transformMat, warpImage.size(), INTER_LINEAR,
		BORDER_CONSTANT, Scalar(0));

		Mat out;
		cv::resize(warpImage, out, Size(charSize, charSize));

		return out;
	}

	// this spatial_ostu algorithm are robust to 
	// the plate which has the same light shine, which is that
	// the light in the left of the plate is strong than the right.
	void spatial_ostu(InputArray _src, int grid_x, int grid_y, Color type) {
		Mat src = _src.getMat();
		int width = src.cols / grid_x;
		int height = src.rows / grid_y;

		// iterate through grid
		for (int i = 0; i < grid_y; i++) {
	    		for (int j = 0; j < grid_x; j++) {
	      			Mat src_cell = Mat(src, Range(i*height, (i + 1)*height), Range(j*width, (j + 1)*width));
	      			if (type == LIGHT) {
					cv::threshold(src_cell, src_cell, 0, 255, CV_THRESH_OTSU + CV_THRESH_BINARY);
	      			}
	      			else if (type == DEEP) {
					cv::threshold(src_cell, src_cell, 0, 255, CV_THRESH_OTSU + CV_THRESH_BINARY_INV);
	      			} 
	    		}
	  	}
	}

	bool clearLiuDing(Mat &img) {
  		std::vector<float> fJump;
  		int whiteCount = 0;
  		const int x = 7;
  		Mat jump = Mat::zeros(1, img.rows, CV_32F);
  		for (int i = 0; i < img.rows; i++) {
    			int jumpCount = 0;

    			for (int j = 0; j < img.cols - 1; j++) {
      				if (img.at<char>(i, j) != img.at<char>(i, j + 1)){
					jumpCount++;
				}

      				if (img.at<uchar>(i, j) == 255) {
        				whiteCount++;
      				}
    			}
    			jump.at<float>(i) = (float) jumpCount;
  		}

  		int iCount = 0;
  		for (int i = 0; i < img.rows; i++) {
    			fJump.push_back(jump.at<float>(i));
    			if (jump.at<float>(i) >= 16 && jump.at<float>(i) <= 45) {
      				// jump condition
      				iCount++;
    			}
  		}

  		// if not is not plate
  		if (iCount * 1.0 / img.rows <= 0.40) {
    			return false;
  		}

  		if (whiteCount * 1.0 / (img.rows * img.cols) < 0.15 || 
			whiteCount * 1.0 / (img.rows * img.cols) > 0.50) {
    				return false;
  		}

  		for (int i = 0; i < img.rows; i++) {
    			if (jump.at<float>(i) <= x) {
				for (int j = 0; j < img.cols; j++) {
					img.at<char>(i, j) = 0;
				}
    			}
  		}

  		return true;
	}
	

	Rect interRect(const Rect& a, const Rect& b) {
  		Rect c;
  		int x1 = a.x > b.x ? a.x : b.x;
  		int y1 = a.y > b.y ? a.y : b.y;
  		c.width = (a.x + a.width < b.x + b.width ? a.x + a.width : b.x + b.width) - x1;
  		c.height = (a.y + a.height < b.y + b.height ? a.y + a.height : b.y + b.height) - y1;
  		c.x = x1;
  		c.y = y1;
  		if (c.width <= 0 || c.height <= 0){
    			c = Rect();
		}
  		return c;
	}

	Rect mergeRect(const Rect& a, const Rect& b) {
  		Rect c;
  		int x1 = a.x < b.x ? a.x : b.x;
  		int y1 = a.y < b.y ? a.y : b.y;
  		c.width = (a.x + a.width > b.x + b.width ? a.x + a.width : b.x + b.width) - x1;
  		c.height = (a.y + a.height > b.y + b.height ? a.y + a.height : b.y + b.height) - y1;
  		c.x = x1;
  		c.y = y1;
  		return c;
	}

	float computeIOU(const Rect& rect1, const Rect& rect2) {

  		Rect inter = interRect(rect1, rect2);
  		Rect urect = mergeRect(rect1, rect2);

  		float iou = (float)inter.area() / (float)urect.area();
 
  		return iou;
	}

	Mat bgrHistogram(const Mat& src)  
	{  
	    	//Separate the BGR channel
		std::vector<Mat> bgr_planes;  
		split(src,bgr_planes);  

		float range[] = { 0, 256 } ;  
		const float* histRange = { range };  
			     
		bool uniform = true; 
		bool accumulate = false;  

		Mat hist1d,normHist1d,hist;  

		for(int i = 0 ;i < 3;i++){  
			calcHist( &bgr_planes[i], 1, 0, Mat(), hist1d, 1, &HISTSIZE, &histRange, uniform, accumulate );  
			normalize(hist1d,hist1d,1.0,0.0,CV_L1);  
			hist.push_back(hist1d);  
		}  

		return hist;  
	}
}

