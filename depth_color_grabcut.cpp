#include <opencv2/opencv.hpp>



int main(){
	cv::Mat color = cv::imread("jesus_color.png");
	cv::Mat depth = cv::imread("jesus_depth.png");
cv::cvtColor(color,color,CV_BGR2GRAY);
cv::cvtColor(color,color,CV_GRAY2BGR);
	cv::Mat filtered;

	cv::namedWindow("color");
	cv::namedWindow("depth");
	imshow("color", color);
	imshow("depth", depth);

	cv::Mat color_region = color.clone();
	cv::rectangle(color_region,cv::Point(200,2),cv::Point(400,410),cv::Scalar(255),1);
	cv::Rect ROI = cv::Rect(cv::Point(200,2),cv::Point(400,410));

	imshow("color", color_region);

	cv::Mat mask,bgdModel,fgdModel;

	cv::grabCut(color, mask, ROI, bgdModel, fgdModel, 1, cv::GC_INIT_WITH_RECT  );
	mask -=2;
	mask.convertTo(mask,CV_8UC1,255);
	cv::namedWindow("color_result");
	imshow("color_result", mask);

	cv::grabCut(depth, mask, ROI, bgdModel, fgdModel, 1, cv::GC_INIT_WITH_RECT  );
	mask -=2;
	mask.convertTo(mask,CV_8UC1,255);
	cv::namedWindow("depth_result");
	imshow("depth_result", mask);

	cv::Mat factor, color_fix;
	depth.convertTo(factor,CV_32FC3,1./255);
	cv::Mat depth_ROI = factor(ROI);

	//	double minVal, maxVal;
	//	minMaxIdx(depth_ROI,  &minVal,  &maxVal);
	//	factor = min(factor,maxVal);
	//	factor = max(factor,minVal);
	//	factor.convertTo(factor,CV_8UC1,255./maxVal);
	//	std::cout << maxVal<<" "<<minVal<<std::endl;


	color.convertTo(color_fix, CV_32FC3);
	color_fix = color_fix.mul(factor).mul(factor).mul(factor);
	color_fix.convertTo(color_fix, CV_8UC3);
	cv::namedWindow("color_fix");
	imshow("color_fix", color_fix);

	cv::grabCut(color_fix, mask, ROI, bgdModel, fgdModel, 1, cv::GC_INIT_WITH_RECT  );
	mask -=2;
	mask.convertTo(mask,CV_8UC1,255);
	cv::namedWindow("color&depth_result");
	imshow("color&depth_result", mask);


	//	input.convertTo(input,CV_32FC1,1./255);
	//	cv::Mat kernel = cv::Mat::ones( 6, 1, CV_32F );
	//	kernel.at<float>(3)=-1;
	//	kernel.at<float>(4)=-1;
	//	kernel.at<float>(5)=-1;
	//	std::cout << kernel<<std::endl;
	//	filter2D(input, filtered, CV_32F, kernel);
	//	imshow("filtered", filtered);
	//
	//
	//	cv::Mat temp = input.clone();
	//	cv::Mat mask,bgdModel,fgdModel;
	//	cv::rectangle(temp,cv::Point(208,19),cv::Point(341,297),cv::Scalar(255),1);
	//	imshow("input", temp);
	//
	//	cv::Rect ROI = cv::Rect(cv::Point(208,19),cv::Point(341,297));
	//	input.convertTo(input,CV_8UC3,255);
	//
	//	cv::grabCut(input, mask, ROI, bgdModel, fgdModel, 4, cv::GC_INIT_WITH_RECT  );
	//	std::cout<< mask<< std::endl;
	//	mask.convertTo(mask,CV_8UC1,255./3);
	//	imshow("result", mask);

	cv::waitKey(0);
	return 1;
}

