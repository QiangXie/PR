#include "segment.hpp"
#include "config.hpp"
#include "character.hpp"
#include "func.hpp"


namespace swpr {
	using namespace std;

	const float DEFAULT_BLUEPERCEMT = 0.3f;
	const float DEFAULT_WHITEPERCEMT = 0.1f;

	CharsSegment::CharsSegment(){
		m_LiuDingSize = DEFAULT_LIUDING_SIZE;
		m_theMatWidth = DEFAULT_MAT_WIDTH;
		m_ColorThreshold = DEFAULT_COLORTHRESHOLD;
		m_BluePercent = DEFAULT_BLUEPERCEMT;
		m_WhitePercent = DEFAULT_WHITEPERCEMT;
		m_debug = DEFAULT_DEBUG;
	}


bool CharsSegment::verifyCharSizes(Mat r) {
  	// Char sizes 45x90
  	float aspect = 45.0f / 90.0f;
  	float charAspect = (float)r.cols / (float)r.rows;
  	float error = 0.7f;
  	float minHeight = 10.f;
  	float maxHeight = 35.f;
  	// We have a different aspect ratio for number 1, and it can be ~0.2
  	float minAspect = 0.05f;
  	float maxAspect = aspect + aspect * error;
  	// area of pixels
  	int area = cv::countNonZero(r);
  	// bb area
  	int bbArea = r.cols * r.rows;
  	//% of pixel in area
  	int percPixels = area / bbArea;

  	if (percPixels <= 1 && charAspect > minAspect && charAspect < maxAspect &&
      		r.rows >= minHeight && r.rows < maxHeight){
    		return true;
	}
  	else{
    		return false;
	}
}


cv::Mat CharsSegment::preprocessChar(cv::Mat in) {
	// Remap image
	int h = in.rows;
	int w = in.cols;

	int charSize = CHAR_SIZE;

	cv::Mat transformMat = cv::Mat::eye(2, 3, CV_32F);
	int m = max(w, h);
	transformMat.at<float>(0, 2) = float(m / 2 - w / 2);
	transformMat.at<float>(1, 2) = float(m / 2 - h / 2);

	cv::Mat warpImage(m, m, in.type());
	warpAffine(in, warpImage, transformMat, warpImage.size(), INTER_LINEAR,
	     BORDER_CONSTANT, Scalar(0));

	cv::Mat out;
	cv::resize(warpImage, out, Size(charSize, charSize));

	return out;
}


//! choose the bese threshold method for chinese
void CharsSegment::judgeChinese(cv::Mat in, cv::Mat& out, Color plateColor) {
	Mat auxRoi = in;
	float valOstu = 1.0, valAdap = 1.0;
	Mat roiOstu, roiAdap;
	bool isChinese = true;

	if (LIGHT == plateColor) {
		threshold(auxRoi, roiOstu, 0, 255, CV_THRESH_BINARY + CV_THRESH_OTSU);
	}
	else if (DEEP == plateColor) {
		threshold(auxRoi, roiOstu, 0, 255, CV_THRESH_BINARY_INV + CV_THRESH_OTSU);
	}
	roiOstu = preprocessChar(roiOstu);

	if (LIGHT == plateColor) {
		adaptiveThreshold(auxRoi, roiAdap, 255, ADAPTIVE_THRESH_MEAN_C, THRESH_BINARY, 3, 0);
	}
	else if (DEEP == plateColor) {
		adaptiveThreshold(auxRoi, roiAdap, 255, ADAPTIVE_THRESH_MEAN_C, THRESH_BINARY_INV, 3, 0);
	}

	roiAdap = preprocessChar(roiAdap);

	if (valOstu >= valAdap) {
		out = roiOstu;
	}
	else {
		out = roiAdap;
	}
}

bool slideChineseWindow(Mat& image, Rect mr, Mat& newRoi, Color plateColor, float slideLengthRatio, bool useAdapThreshold) {
	std::vector<CCharacter> charCandidateVec;

	Rect maxrect = mr;
	Point tlPoint = mr.tl();

	bool isChinese = true;
	int slideLength = int(slideLengthRatio * maxrect.width);
	int slideStep = 1;
	int fromX = 0;
	fromX = tlPoint.x;

	for (int slideX = -slideLength; slideX < slideLength; slideX += slideStep) {
		float x_slide = 0;

		x_slide = float(fromX + slideX);

		float y_slide = (float)tlPoint.y;
		Point2f p_slide(x_slide, y_slide);

		//cv::circle(image, p_slide, 2, Scalar(255), 1);

		int chineseWidth = int(maxrect.width);
		int chineseHeight = int(maxrect.height);

		Rect rect(Point2f(x_slide, y_slide), Size(chineseWidth, chineseHeight));

		if (rect.tl().x < 0 || rect.tl().y < 0 || rect.br().x >= image.cols || rect.br().y >= image.rows){
			continue;
		}

		Mat auxRoi = image(rect);

		Mat roiOstu, roiAdap;

		if (LIGHT == plateColor) {
			threshold(auxRoi, roiOstu, 0, 255, CV_THRESH_BINARY + CV_THRESH_OTSU);
		}
		else if (DEEP == plateColor) {
			threshold(auxRoi, roiOstu, 0, 255, CV_THRESH_BINARY_INV + CV_THRESH_OTSU);
		}
		roiOstu = preprocessChar(roiOstu, kChineseSize);

		CCharacter charCandidateOstu;
		charCandidateOstu.setCharacterPos(rect);
		charCandidateOstu.setCharacterMat(roiOstu);
		charCandidateOstu.setIsChinese(isChinese);
		charCandidateVec.push_back(charCandidateOstu);

		if (useAdapThreshold) {
			if (LIGHT == plateColor) {
				adaptiveThreshold(auxRoi, roiAdap, 255, ADAPTIVE_THRESH_MEAN_C, THRESH_BINARY, 3, 0);
			}
			else if (DEEP == plateColor) {
				adaptiveThreshold(auxRoi, roiAdap, 255, ADAPTIVE_THRESH_MEAN_C, THRESH_BINARY_INV, 3, 0);
			}
			roiAdap = preprocessChar(roiAdap, kChineseSize);

			CCharacter charCandidateAdap;
			charCandidateAdap.setCharacterPos(rect);
			charCandidateAdap.setCharacterMat(roiAdap);
			charCandidateAdap.setIsChinese(isChinese);
			charCandidateVec.push_back(charCandidateAdap);
		}
	}

	double overlapThresh = 0.1;
	NMStoCharacter(charCandidateVec, overlapThresh);

	if (charCandidateVec.size() >= 1) {
		std::sort(charCandidateVec.begin(), charCandidateVec.end(),
		[](const CCharacter& r1, const CCharacter& r2) {
		return r1.getCharacterScore() > r2.getCharacterScore();
	});

		newRoi = charCandidateVec.at(0).getCharacterMat();
		return true;
	}

	return false;
}

Color CharsSegment::getPlateColor(Mat greyImage){
	double pixSum = 0;
	int counterMoreThanAveragePix = 0;
	int counterLessThanAveragePix = 0;
	int nr = greyImage.rows;
	int nl = greyImage.cols*greyImage.channels();	

	for(int i = 0; i < nr; ++i){
		const uchar* data = greyImage.ptr<uchar>(i);
		for(int j = 0; j < nl; ++j){
			pixSum += data[j]; 
		}
	}
	double pixAverage = pixSum / (greyImage.rows * greyImage.cols * greyImage.channels());

	for(int i = 0; i < nr; ++i){
		const uchar* data = greyImage.ptr<uchar>(i);
		for(int j = 0; j < nl; ++j){
			if(data[j] > pixAverage){
				counterMoreThanAveragePix++;
			}
			else{
				counterLessThanAveragePix++;
			}

		}
	}
	if(counterLessThanAveragePix < counterMoreThanAveragePix){	
		return DEEP;
	}
	else{
		return LIGHT;
	}
}

int CharsSegment::charsSegment(Mat input, std::vector<Mat>& resultVec) {
	if (!input.data) {
	  	LOG(FATAL) << "Error:Image to segment is empty.";
	}

  	Mat inputGrey;
  	cvtColor(input, inputGrey, CV_BGR2GRAY);
  	Color plateColor = this->getPlateColor(inputGrey);

	Mat imgThreshold = inputGrey.clone();
	spatial_ostu(imgThreshold, 1, 1, plateColor);

	// remove liuding and hor lines
	// also judge weather is plate use jump count

	if (!clearLiuDing(imgThreshold)) {
		return 0x02;
	}
	clearLiuDing(imgThreshold);

	Mat imgContours;
	imgThreshold.copyTo(imgContours);

	std::vector<std::vector<Point> > contours;
	findContours(imgContours,
	       contours,               // a vector of contours
	       CV_RETR_EXTERNAL,       // retrieve the external contours
	       CV_CHAIN_APPROX_NONE);  // all pixels of each contours

	vector<vector<Point> >::iterator itc = contours.begin();
	vector<Rect> vecRect;

	while (itc != contours.end()) {
		Rect mr = boundingRect(Mat(*itc));
		Mat auxRoi(imgThreshold, mr);

		if (verifyCharSizes(auxRoi)) {
			vecRect.push_back(mr);
		}
		++itc;
	}

	if (vecRect.size() == 0) {
		return 0x03;
	}

	vector<Rect> sortedRect(vecRect);
  	std::sort(sortedRect.begin(), sortedRect.end(),
            [](const Rect& r1, const Rect& r2) { return r1.x < r2.x; });

	size_t specIndex = 0;

	specIndex = GetSpecificRect(sortedRect);

	Rect chineseRect;
	if (specIndex < sortedRect.size()){
	    chineseRect = GetChineseRect(sortedRect[specIndex]);
	}
	else{
		return 0x04;
	}

	vector<Rect> newSortedRect;
	newSortedRect.push_back(chineseRect);
	RebuildRect(sortedRect, newSortedRect, specIndex);

	if (newSortedRect.size() == 0) {
		return 0x05;
	}

	bool useSlideWindow = true;
	bool useAdapThreshold = true;
	//bool useAdapThreshold = CParams::instance()->getParam1b();
	
	for (size_t i = 0; i < newSortedRect.size(); i++) {
		Rect mr = newSortedRect[i];

		// Mat auxRoi(img_threshold, mr);
		Mat auxRoi(inputGrey, mr);
		Mat newRoi;
		//Mat newRoi(input,mr);

		if (i == 0) {
			if (useSlideWindow) {
				float slideLengthRatio = 0.1f;
				//float slideLengthRatio = CParams::instance()->getParam1f();
				if (!slideChineseWindow(inputGrey, mr, newRoi, plateColor, slideLengthRatio, useAdapThreshold))
				  judgeChinese(auxRoi, newRoi, plateColor);
			}
			else{
				judgeChinese(auxRoi, newRoi, plateColor);
			}
		}
		else {
			if (LIGHT == plateColor) {  
				threshold(auxRoi, newRoi, 0, 255, CV_THRESH_BINARY + CV_THRESH_OTSU);
			}
			else if (DEEP == plateColor) {
				threshold(auxRoi, newRoi, 0, 255, CV_THRESH_BINARY_INV + CV_THRESH_OTSU);
			}
			newRoi = preprocessChar(newRoi);
		}

		 resultVec.push_back(newRoi);
	}
	if(SAVE_SEGMENT_FLAG){	
		if(resultVec.size() != 7){
			std::time_t now = std::time(nullptr);
			std::stringstream ssImName;
			ssImName << "./segment_fault/" << now << ".jpg";
			std::string imName = ssImName.str();

			Mat imgToSave;
			Mat3b inputGreyBGR,imgThresholdBGR;
			cvtColor(inputGrey,inputGreyBGR,COLOR_GRAY2BGR);
			cvtColor(imgThreshold,imgThresholdBGR,COLOR_GRAY2BGR);


			int cols = inputGrey.cols;
			int rows = inputGrey.rows * 4;
			imgToSave.create(rows,cols,inputGreyBGR.type());
			//ori img
			input.copyTo(imgToSave(Rect(0, 0, inputGrey.cols, inputGrey.rows)));
			//greyscale img
			inputGreyBGR.copyTo(imgToSave(Rect(0, inputGrey.rows, imgThreshold.cols, imgThreshold.rows)));
			//bin img
			imgThresholdBGR.copyTo(imgToSave(Rect(0, inputGrey.rows*2, imgThreshold.cols, imgThreshold.rows)));
			for(int i = 0; i < newSortedRect.size();++i){
				rectangle(imgThresholdBGR,newSortedRect[i],Scalar(0,255,0),1);
			}
			imgThresholdBGR.copyTo(imgToSave(Rect(0, inputGrey.rows*3, imgThreshold.cols, imgThreshold.rows)));
			
			imwrite(imName,imgToSave);	
		}	
	}
  	return 0;
}


Rect CharsSegment::GetChineseRect(const Rect rectSpe) {
	int height = rectSpe.height;
	float newwidth = rectSpe.width * 1.15f;
	int x = rectSpe.x;
	int y = rectSpe.y;

	int newx = x - int(newwidth * 1.15);
	newx = newx > 0 ? newx : 0;

	Rect a(newx, y, int(newwidth), height);

	return a;
}

int CharsSegment::GetSpecificRect(const vector<Rect>& vecRect) {
	vector<int> xpositions;
	int maxHeight = 0;
	int maxWidth = 0;

	for (size_t i = 0; i < vecRect.size(); i++) {
		xpositions.push_back(vecRect[i].x);

		if (vecRect[i].height > maxHeight) {
			maxHeight = vecRect[i].height;
		}
		if (vecRect[i].width > maxWidth) {
			maxWidth = vecRect[i].width;
		}
	}

	int specIndex = 0;
	for (size_t i = 0; i < vecRect.size(); i++) {
		Rect mr = vecRect[i];
		int midx = mr.x + mr.width / 2;

		// use known knowledage to find the specific character
		// position in 1/7 and 2/7
		if ((mr.width > maxWidth * 0.8 || mr.height > maxHeight * 0.8) &&
		  (midx < int(m_theMatWidth / 7) * 2 &&
		  midx > int(m_theMatWidth / 7) * 1)) {
			specIndex = i;
		}
	}

  	return specIndex;
}

int CharsSegment::RebuildRect(const vector<Rect>& vecRect,vector<Rect>& outRect, int specIndex) {
	int count = 6;
	for (size_t i = specIndex; i < vecRect.size() && count; ++i, --count) {
		outRect.push_back(vecRect[i]);
	}

	return 0;
}

}
