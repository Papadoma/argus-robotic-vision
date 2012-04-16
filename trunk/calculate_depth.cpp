#include "opencv2/calib3d/calib3d.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"

#include <vector>
#include <string>
#include <algorithm>
#include <iostream>
#include <iterator>
#include <stdio.h>
#include <stdlib.h>
#include <ctype.h>

using namespace cv;
using namespace std;

StereoSGBM sgbm;

void print_all(){
	printf("\n\n\n\n\n\n\n\n\n\n");
	printf("Use the following keys to change camera parameters:\n");
	printf(	"\t'+' - increase parameter\n");
	printf(	"\t'-' - decrease parameter\n");
	printf(	"\t'q' [%d]- preFilterCap\n",sgbm.preFilterCap);
	printf(	"\t'w' [%d]- SADWindowSize\n",sgbm.SADWindowSize);
	printf(	"\t'e' [%d]- P1   P2>P1\n",sgbm.P1);
	printf(	"\t'r' [%d]- P2\n",sgbm.P2);
	printf(	"\t't' [%d]- minDisparity\n",sgbm.minDisparity);
	printf(	"\t'y' [%d]- numberOfDisparities\n",sgbm.numberOfDisparities);
	printf(	"\t'a' [%d]- uniquenessRatio   5-15 is usually good\n",sgbm.uniquenessRatio);
	printf(	"\t's' [%d]- speckleWindowSize   0 disables filtering\n",sgbm.speckleWindowSize);
	printf(	"\t'd' [%d]- speckleRange   16 or 32 is normall good\n",sgbm.speckleRange);
	printf(	"\t'f' [%d]- disp12MaxDiff  negative values disables this!\n",sgbm.disp12MaxDiff);

	return;
}

void IncParam(int param){
	switch(param){
	case 1:	sgbm.preFilterCap++; break;
	case 2:sgbm.SADWindowSize=sgbm.SADWindowSize+2;sgbm.P2 = 32*sgbm.SADWindowSize*sgbm.SADWindowSize;sgbm.P1 = 8*sgbm.SADWindowSize*sgbm.SADWindowSize;break;
	case 3:sgbm.P1++;break;
	case 4:sgbm.P2++;break;
	case 5:sgbm.minDisparity++;break;
	case 6:sgbm.numberOfDisparities=sgbm.numberOfDisparities+16;break;
	case 7:sgbm.uniquenessRatio++;break;
	case 8:sgbm.speckleWindowSize++;break;
	case 9:sgbm.speckleRange++;break;
	case 10:sgbm.disp12MaxDiff++;break;
	}
	return;
}

void DecParam(int param){
	switch(param){
	  case 1:(sgbm.preFilterCap>0?sgbm.preFilterCap--:sgbm.preFilterCap=0); break;
	  case 2:(sgbm.SADWindowSize>1?sgbm.SADWindowSize=sgbm.SADWindowSize-2:sgbm.SADWindowSize=1);sgbm.P2 = 32*sgbm.SADWindowSize*sgbm.SADWindowSize;sgbm.P1 = 8*sgbm.SADWindowSize*sgbm.SADWindowSize;break;
	  case 3:(sgbm.P1>0?sgbm.P1--:sgbm.P1=0);break;
	  case 4:(sgbm.P2>0?sgbm.P2--:sgbm.P2=0);break;
	  case 5:(sgbm.minDisparity>0?sgbm.minDisparity--:sgbm.minDisparity=0);break;
	  case 6:(sgbm.numberOfDisparities>16?sgbm.numberOfDisparities=sgbm.numberOfDisparities-16:sgbm.numberOfDisparities=16);break;
	  case 7:(sgbm.uniquenessRatio>0?sgbm.uniquenessRatio--:sgbm.uniquenessRatio=0);break;
	  case 8:(sgbm.speckleWindowSize>0?sgbm.speckleWindowSize--:sgbm.speckleWindowSize=0);break;
	  case 9:(sgbm.speckleRange>0?sgbm.speckleRange--:sgbm.speckleRange=0);break;
	  case 10:(sgbm.disp12MaxDiff>-1?sgbm.disp12MaxDiff--:sgbm.disp12MaxDiff=-1);break;
	}
	return;
}

int main()
{
  
	string file_left="/home/papadoma/argussvn/trunk/bumble_calib/cal_left01.ppm";
	string file_right="/home/papadoma/argussvn/trunk/bumble_calib/cal_right01.ppm";
	string intrinsics="/home/papadoma/argussvn/trunk/bumble_calib/intrinsics_bumblebee.yml";
	string extrinsics="/home/papadoma/argussvn/trunk/bumble_calib/extrinsics_bumblebee.yml";
	
	
	Mat image_left = imread(file_left,0);
	Mat image_right = imread(file_left,0);
	Mat image_left_calibrated;
	Mat image_right_calibrated;

	//cvNamedWindow("left",CV_WINDOW_AUTOSIZE);
	//cvNamedWindow("right",CV_WINDOW_AUTOSIZE);

	cvNamedWindow("left_calibrated",CV_WINDOW_AUTOSIZE);
	cvNamedWindow("right_calibrated",CV_WINDOW_AUTOSIZE);

	cvNamedWindow("depth",CV_WINDOW_AUTOSIZE);

	//imshow("left",image_left);
	//imshow("right",image_right);

	Mat rmap[2][2];
	Mat cameraMatrix[2], distCoeffs[2];
	Mat R, T, E, F;
	Mat R1, R2, P1, P2, Q;
	Size imageSize=image_left.size();

	FileStorage fs(intrinsics, CV_STORAGE_READ);
	if( fs.isOpened() )
	{
		fs["M1"] >> cameraMatrix[0];
		fs["D1"] >> distCoeffs[0] ;
		fs["M2"] >> cameraMatrix[1];
		fs["D2"] >> distCoeffs[1] ;
		fs.release();
	}
	else
		cout << "Error: can not load the intrinsic parameters\n";

	fs.open(extrinsics, CV_STORAGE_READ);
	if( fs.isOpened() )
	{
		fs["R"] >> R;
		fs["T"] >> T;
		fs["R1"] >> R1;
		fs["R2"] >> R2;
		fs["P1"] >> P1;
		fs["P2"] >> P2;
		fs["Q"] >> Q;

		fs.release();
	}
	else
		cout << "Error: can not load the extrinsics parameters\n";


	initUndistortRectifyMap(cameraMatrix[0], distCoeffs[0], R1, P1, imageSize, CV_16SC2, rmap[0][0], rmap[0][1]);
	initUndistortRectifyMap(cameraMatrix[1], distCoeffs[1], R2, P2, imageSize, CV_16SC2, rmap[1][0], rmap[1][1]);

	remap(image_left, image_left_calibrated, rmap[0][0], rmap[0][1], CV_INTER_LINEAR);
	remap(image_right, image_right_calibrated, rmap[1][0], rmap[1][1], CV_INTER_LINEAR);

	imshow("left_calibrated",image_left_calibrated);
	imshow("right_calibrated",image_right_calibrated);

	Mat* imageDepth=new Mat();
	Mat* imageDepthNormalized=new Mat();


	//StereoSGBM BM_comp(minDisparity, numDisparities, SADWindowSize,	p1, p2, disp12MaxDiff, preFilterCap, uniquenessRatio, speckleWindowSize, speckleRange, fullDP);

	sgbm.preFilterCap = 63;
	sgbm.SADWindowSize = 1;
	sgbm.P1 = 8*sgbm.SADWindowSize*sgbm.SADWindowSize;
	sgbm.P2 = 32*sgbm.SADWindowSize*sgbm.SADWindowSize;
	sgbm.minDisparity = 0;
	sgbm.numberOfDisparities = 16;
	sgbm.uniquenessRatio = 0;
	sgbm.speckleWindowSize = 100;
	sgbm.speckleRange = 32;
	sgbm.disp12MaxDiff = 1;
	sgbm.fullDP = true;
	sgbm(image_left_calibrated,image_right_calibrated,*imageDepth);
	imageDepth->convertTo(*imageDepth, CV_8U);
	//cvFindStereoCorrespondenceBM(&image_left_calibrated, &image_right_calibrated, imageDepth, BMState);


	imshow("depth",*imageDepth);

	int key=0;

	int param=0;
	while((key = (cvWaitKey(10) & 255)) !=  0x1b)
	{

		sgbm(image_left_calibrated,image_right_calibrated,*imageDepth);
		imageDepth->convertTo(*imageDepth, CV_8U);
		imshow("depth",*imageDepth);
		//printf("\n%d",key);
		switch(key)
		{
		case 0x71:	param=1;print_all();		break;
		case 0x77:	param=2;print_all();		break;
		case 0x65:	param=3;print_all();		break;
		case 0x72:	param=4;print_all();		break;
		case 0x74:	param=5;print_all();		break;
		case 0x79:	param=6;print_all();		break;
		case 0x61:	param=7;print_all();		break;
		case 0x73:	param=8;print_all();		break;
		case 0x64:	param=9;print_all();		break;
		case 0x66:	param=10;print_all();		break;
		case 0xAB:	IncParam(param);print_all();	break;
		case 0xAD:	DecParam(param);print_all();	break;
		}
	}

	//cvDestroyWindow("left");
	//cvDestroyWindow("right");
	cvDestroyWindow("depth");
	cvDestroyWindow("left_calibrated");
	cvDestroyWindow("right_calibrated");
	return 0;
}

