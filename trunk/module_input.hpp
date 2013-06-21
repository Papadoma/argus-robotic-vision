/*
 * module_input.cpp
 *
 * This file is part of my final year's project for the Department
 * of Electrical and Computer Engineering of Aristotle University
 * of Thessaloniki, 2013.
 *
 * Author:	Miltiadis-Alexios Papadopoulos
 *
 */

#pragma once
#ifndef MODULE_EYE_HPP
#define MODULE_EYE_HPP
#define USE_CL_DRIVER true

#include <opencv2/opencv.hpp>
#include <opencv/cv.h>

#if (defined(_WIN32) || defined(__WIN32__) || defined(WIN32)) && USE_CL_DRIVER
#include <CLEyeMulticam.h>
#endif

class module_eye{
private:
#if (defined(_WIN32) || defined(__WIN32__) || defined(WIN32)) && USE_CL_DRIVER
	typedef CLEyeCameraInstance capture_type;
#else
	typedef cv::VideoCapture capture_type;
#endif

	capture_type capture_left;
	capture_type capture_right;
	bool EoF;
	bool use_camera;
	int width;	//Frame width
	int height; //Frame height

	cv::VideoCapture file_left;
	cv::VideoCapture file_right;

#if (defined(_WIN32) || defined(__WIN32__) || defined(WIN32)) && USE_CL_DRIVER
	PBYTE pCapBufferLeft;
	PBYTE pCapBufferRight;
	IplImage *pCapImageLeft;
	IplImage *pCapImageRight;
#endif

public:
	module_eye();
	module_eye(std::string, std::string);
	~module_eye();
	bool getFrame(cv::Mat&, cv::Mat&);
	cv::Size getSize();

};

#endif
