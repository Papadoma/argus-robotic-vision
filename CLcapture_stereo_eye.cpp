//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//
// This file is part of CL-EyeMulticam SDK
//
// C++ CLEyeFaceTracker Sample Application
//
// For updates and file downloads go to: http://codelaboratories.com
//
// Copyright 2008-2010 (c) Code Laboratories, Inc. All rights reserved.
//
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
#include "stdafx.h"
#include <vector>
using namespace std;

// Sample camera capture and processing class
class CLEyeStereoCameraCapture
{
	CHAR _windowName[256];
	CHAR _depthWindowName[256];
	GUID _cameraGUID[2];
	CLEyeCameraInstance _cam[2];
	CLEyeCameraColorMode _mode;
	CLEyeCameraResolution _resolution;
	float _fps;
	HANDLE _hThread;
	bool _running;

public:
	CLEyeStereoCameraCapture(CLEyeCameraResolution resolution, float fps) :
		_mode(CLEYE_MONO_RAW), _resolution(resolution), _fps(fps), _running(false)
	{
		strcpy(_windowName, "Capture Window");
		strcpy(_depthWindowName, "Stereo Depth");
		for(int i = 0; i < 2; i++)
			_cameraGUID[i] = CLEyeGetCameraUUID(i);
	}
	bool StartCapture()
	{
		_running = true;
		cvNamedWindow(_windowName, CV_WINDOW_AUTOSIZE);
		cvNamedWindow(_depthWindowName, CV_WINDOW_AUTOSIZE);
		// Start CLEye image capture thread
		_hThread = CreateThread(NULL, 0, &CLEyeStereoCameraCapture::CaptureThread, this, 0, 0);
		if(_hThread == NULL)
		{
			MessageBox(NULL,"Could not create capture thread","CLEyeMulticamTest", MB_ICONEXCLAMATION);
			return false;
		}
		return true;
	}
	void StopCapture()
	{
		if(!_running)	return;
		_running = false;
		WaitForSingleObject(_hThread, 1000);
		cvDestroyWindow(_windowName);
		cvDestroyWindow(_depthWindowName);
	}
	void IncrementCameraParameter(int param)
	{
		for(int i = 0; i < 2; i++)
		{
			if(!_cam[i])	continue;
			CLEyeSetCameraParameter(_cam[i], (CLEyeCameraParameter)param, CLEyeGetCameraParameter(_cam[i], (CLEyeCameraParameter)param)+10);
		}
	}
	void DecrementCameraParameter(int param)
	{
		for(int i = 0; i < 2; i++)
		{
			if(!_cam[i])	continue;
			CLEyeSetCameraParameter(_cam[i], (CLEyeCameraParameter)param, CLEyeGetCameraParameter(_cam[i], (CLEyeCameraParameter)param)-10);
		}
	}
	void Run()
	{
		int w, h;
		IplImage *pCapImage[2];
		IplImage *pDisplayImage;

		// Create camera instances
		for(int i = 0; i < 2; i++)
		{
			_cam[i] = CLEyeCreateCamera(_cameraGUID[i], _mode, _resolution, _fps);
			if(_cam[i] == NULL)	return;
			// Get camera frame dimensions
			CLEyeCameraGetFrameDimensions(_cam[i], w, h);
			// Create the OpenCV images
			pCapImage[i] = cvCreateImage(cvSize(w, h), IPL_DEPTH_8U, 1);

			// Set some camera parameters
			CLEyeSetCameraParameter(_cam[i], CLEYE_GAIN, 0);
			CLEyeSetCameraParameter(_cam[i], CLEYE_EXPOSURE, 511);

			// Start capturing
			CLEyeCameraStart(_cam[i]);
		}
		pDisplayImage = cvCreateImage(cvSize(w*2, h), IPL_DEPTH_8U, 1);

		// Get the current app path
		char strPathName[_MAX_PATH];
		GetModuleFileName(NULL, strPathName, _MAX_PATH);
		*(strrchr(strPathName, '\\') + 1) = '\0';

		Init(w, h);

		// image capturing loop
		while(_running)
		{
			PBYTE pCapBuffer;
			// Capture camera images
			for(int i = 0; i < 2; i++)
			{
				cvGetImageRawData(pCapImage[i], &pCapBuffer);
				CLEyeCameraGetFrame(_cam[i], pCapBuffer, (i==0)?2000:0);
			}



			// Display stereo image
			for(int i = 0; i < 2; i++)
			{
				cvSetImageROI(pDisplayImage, cvRect(i*w, 0, w, h));
				cvCopy(pCapImage[i], pDisplayImage);
			}
			cvResetImageROI(pDisplayImage);

			cvShowImage(_windowName, pDisplayImage);
		}

		for(int i = 0; i < 2; i++)
		{
			// Stop camera capture
			CLEyeCameraStop(_cam[i]);
			// Destroy camera object
			CLEyeDestroyCamera(_cam[i]);
			// Destroy the allocated OpenCV image
			cvReleaseImage(&pCapImage[i]);
			_cam[i] = NULL;
		}
	}
	static DWORD WINAPI CaptureThread(LPVOID instance)
	{
		// seed the RNG with current tick count and thread id
		srand(GetTickCount() + GetCurrentThreadId());
		// forward thread to Capture function
		CLEyeStereoCameraCapture *pThis = (CLEyeStereoCameraCapture *)instance;
		pThis->Run();
		return 0;
	}

	int cornersX, cornersY, cornersN;
	int sampleCount;
	bool calibrationStarted;
	bool calibrationDone;

	CvSize imageSize;
	int imageWidth;
	int imageHeight;

	vector<CvPoint2D32f> ponintsTemp[2];
	vector<CvPoint3D32f> objectPoints;
	vector<CvPoint2D32f> points[2];
	vector<int> npoints;

	int image_num;

	void Init(int imageWidth, int imageHeight)
	{

		imageSize = cvSize(imageWidth, imageHeight);
		calibrationStarted = false;
		calibrationDone = false;
		image_num=1;
		sampleCount = 0;
	}


	void TakeSnapshot(){
		IplImage *pCapImage[2];


		PBYTE pCapBuffer;
		int w,h;

		for(int i = 0; i < 2; i++)
		{
			CLEyeCameraGetFrameDimensions(_cam[i], w, h);
			pCapImage[i] = cvCreateImage(cvSize(w, h), IPL_DEPTH_8U, 1);
			cvGetImageRawData(pCapImage[i], &pCapBuffer);
			CLEyeCameraGetFrame(_cam[i], pCapBuffer, (i==0)?2000:0);
		}


		std::stringstream ss1;
		std::string str1;
		ss1 <<"cal_left"<< image_num<<".jpg";
		ss1 >> str1;
		cvSaveImage(str1.c_str(),pCapImage[0]);

		std::stringstream ss2;
		std::string str2;
		ss2 <<"cal_right"<< image_num<<".jpg";
		ss2 >> str2;
		cvSaveImage(str2.c_str() ,pCapImage[1]);

		image_num++;
		return;
	}
};

/*int _tmain(int argc, _TCHAR* argv[])
{
	printf("Use the following keys to change camera parameters:\n"
			"\t'g' - select gain parameter\n"
			"\t'e' - select exposure parameter\n"
			"\t'+' - increment selected parameter\n"
			"\t'-' - decrement selected parameter\n"
			"\t'c' - take snapshot\n");

	CLEyeStereoCameraCapture *cam = NULL;
	// Query for number of connected cameras
	int numCams = CLEyeGetCameraCount();
	if(numCams < 2)
	{
		printf("No PS3Eye cameras detected\n");
		return -1;
	}
	// Create camera capture object
	cam = new CLEyeStereoCameraCapture(CLEYE_VGA, 60);
	printf("Starting capture\n");
	cam->StartCapture();


	// The <ESC> key will exit the program
	CLEyeStereoCameraCapture *pCam = NULL;
	int param = -1, key;
	while((key = cvWaitKey(0)) != 0x1b)
	{
		switch(key)
		{
		case 'g':	case 'G':	printf("Parameter Gain\n");		param = CLEYE_GAIN;		break;
		case 'e':	case 'E':	printf("Parameter Exposure\n");	param = CLEYE_EXPOSURE;	break;
		case '+':	if(cam)		cam->IncrementCameraParameter(param);					break;
		case '-':	if(cam)		cam->DecrementCameraParameter(param);					break;
		case 'c':	if(cam)		cam->TakeSnapshot();									break;
		}
	}
	cam->StopCapture();
	delete cam;
	return 0;
}*/

