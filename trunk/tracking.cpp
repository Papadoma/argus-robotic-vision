#include "module_input.hpp"

struct tracking_objects{
	bool on_tracking;
	cv::Rect bounding_rect;
	cv::Point3d center;
	cv::Mat mask;
	cv::MatND hist;
};

class tracking{
private:
	module_eye test;
	std::vector<cv::Point> mask_points;
	cv::Mat left,right;
	cv::Mat mask_frame,tracking_view;
	int width,height;

	cv::Rect bounding_rect;
	std::vector<tracking_objects> objects;

	int H_value_max;
	int H_value_min;
	void threshold_image();

	cv::KalmanFilter KF;
public:

	tracking();
	void draw_ROI();
	void refresh_frame();
	void refresh_window();
	void track_blob();
	void find_blob();
};


tracking::tracking(){

	H_value_max = 255;
	H_value_min = 0;


	mask_frame = cv::Mat::zeros(test.getSize(), CV_8UC3);
	tracking_view = cv::Mat::zeros(test.getSize(), CV_8UC3);

	height = test.getSize().height;
	width = test.getSize().width;
}
void tracking::threshold_image(){
	cvtColor(left, mask_frame, CV_BGR2HSV);
	cv::inRange(mask_frame, cv::Scalar(0, 80, 50), cv::Scalar(10, 255, 255), mask_frame);
	cv::Mat element = cv::getStructuringElement( cv::MORPH_ELLIPSE , cv::Size(7,7), cv::Point( 3, 3 ) );
	//morphologyEx( mask_frame, mask_frame, cv::MORPH_CLOSE  , element);
	//imshow("mask_frame",mask_frame);
	cv::erode(mask_frame,mask_frame,cv::Mat(),cv::Point(),2);
	cv::dilate(mask_frame,mask_frame,cv::Mat(),cv::Point(),10);
}

void tracking::find_blob(){
	objects.clear();
	cv::Rect clear_rect= cv::Rect(0,0,width,height);
	threshold_image();

	std::vector<std::vector<cv::Point> > contours;
	std::vector<cv::Point> joined_contours;
	std::vector<cv::Vec4i> hierarchy;
	cv::findContours( mask_frame, contours, hierarchy,CV_RETR_EXTERNAL , CV_CHAIN_APPROX_NONE );

	for( int i = 0; i < (int)contours.size(); i++ ){
		tracking_objects obj;
		obj.on_tracking = false;
		cv::Rect bounding_rect = clear_rect & boundingRect(contours[i]);
		obj.bounding_rect = bounding_rect;

		cv::Mat mask, bgdModel,fgdModel;
		grabCut(left, mask, bounding_rect,  bgdModel,  fgdModel, 1, cv::GC_INIT_WITH_RECT  );

		threshold(mask, mask, 2, 255, cv::THRESH_BINARY);
		drawContours( mask_frame, contours, i, cv::Scalar(255), CV_FILLED, 8 ); //Fill biggest blob
		//sometimes, grabcut doesnt work
		if(countNonZero(mask)){
			obj.mask = mask(bounding_rect).clone();
		}else{
			obj.mask = mask_frame(bounding_rect).clone();
		}

		obj.center = cv::Point3d(cvRound(obj.bounding_rect.x+obj.bounding_rect.width/2),cvRound(obj.bounding_rect.y+obj.bounding_rect.height/2),0);

		cv::Mat hsv;
		cv::Mat obj_img = left(obj.bounding_rect);
		cvtColor(obj_img, hsv, CV_BGR2HSV);

		int hbins = 30, sbins = 32, vbins = 32;
		int histSize[] = {hbins, sbins, vbins};
		// hue varies from 0 to 179, see cvtColor
		float hranges[] = { 0, 180 };
		// saturation varies from 0 (black-gray-white) to
		// 255 (pure spectrum color)
		float sranges[] = { 0, 256 };
		float vranges[] = { 0, 256 };
		const float* ranges[] = { hranges, sranges, vranges };

		int channels[] = {0, 1, 2};

		calcHist( &hsv, 1, channels, obj.mask,obj.hist, 2, histSize, ranges,true, false );

		objects.push_back(obj);
	}
	std::cout << "Found: " << contours.size() << "blobs" << std::endl;
}

void tracking::track_blob(){
	float hranges[] = { 0, 180 };
	float sranges[] = { 0, 256 };
	float vranges[] = { 0, 256 };
	const float* ranges[] = { hranges, sranges, vranges };
	int channels[] = {0, 1, 2};
	cv::Mat hsv;
	cvtColor(left, hsv, CV_BGR2HSV);
	cv::TermCriteria criteria;

	for(int i=0;i<(int)objects.size();i++){
		cv::MatND backproj;
		cv::RotatedRect detected_rect;
		cv::calcBackProject( &hsv, 1, channels, objects[i].hist, backproj, ranges, 1, true );

imshow("backproj",backproj);
		//detected_rect = CamShift(backproj, objects[i].bounding_rect, cv::TermCriteria( CV_TERMCRIT_EPS | CV_TERMCRIT_ITER, 1, 1 ));
		//objects[i].bounding_rect=detected_rect.boundingRect();
		meanShift(backproj, objects[i].bounding_rect, cv::TermCriteria( CV_TERMCRIT_EPS | CV_TERMCRIT_ITER, 10, 1 ));
		 ellipse( tracking_view, detected_rect, cv::Scalar(0,0,255), 3, CV_AA );

	}
}

void tracking::draw_ROI(){
	for(int i =0; i<(int) objects.size();i++){
		//		cv::Mat color_mask = objects[i].mask.clone();
		//		cv::cvtColor(color_mask,color_mask,CV_GRAY2BGR);
		//		tracking_view(objects[i].bounding_rect) = tracking_view(objects[i].bounding_rect) + objects[i].mask;
		cv::Mat dest = tracking_view(objects[i].bounding_rect);
		cv::Mat color_mask = objects[i].mask.clone();
		cv::cvtColor(color_mask,color_mask,CV_GRAY2BGR);
		add(dest,  color_mask, dest);
		rectangle(tracking_view,objects[i].bounding_rect, cv::Scalar(0,255,0), 2);
		//cv::Point center2d = cv::Point(objects[i].center.x,objects[i].center.y);
		//circle(tracking_view, center2d, 2, cv::Scalar(0,255,0),-1);
	}
}

void tracking::refresh_frame(){
	test.getFrame(left, right);
	tracking_view = left.clone();
	draw_ROI();
}

void tracking::refresh_window(){
	cv::Mat imgResult(height,2*width,CV_8UC3); // Your final imageasd
	cv::Mat roiImgResult_Left = imgResult(cv::Rect(0,0,left.cols,left.rows));
	cv::Mat roiImgResult_Right = imgResult(cv::Rect(right.cols,0,right.cols,right.rows));
	cv::Mat roiImg1 = (left)(cv::Rect(0,0,left.cols,left.rows));
	cv::Mat roiImg2 = (right)(cv::Rect(0,0,right.cols,right.rows));
	roiImg1.copyTo(roiImgResult_Left);
	roiImg2.copyTo(roiImgResult_Right);

	imshow( "Camera", imgResult );
	imshow("tracking red",tracking_view);

}




int main(){
	tracking inst;

	while(1){

		inst.refresh_frame();
		inst.track_blob();
		inst.refresh_window();

		int key_pressed = cvWaitKey(1) & 255;
		if ( key_pressed == 27 ) break;
		if ( key_pressed == 13 ) inst.find_blob();
		//
	}
	return 0;
}
