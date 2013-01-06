#include <opencv.hpp>
#include <stdio.h>

#include "module_file.hpp"

using namespace std;
using namespace cv;

module_file::module_file(){
	//	filename_left="C:/Users/papadoma/Videos/youtube3D_left.mpg";
	//	filename_right="C:/Users/papadoma/Videos/youtube3D_right.mpg";
	filename_left="left1.mpg";
	filename_right="right1.mpg";

	EoF = false;
	cout<<"Creating capture instances\n";
	capture_left = new VideoCapture();
	capture_right = new VideoCapture();
	if (capture_left&&capture_right)
	{
		cout << "Created capture instances!\n";
	}else{
		cout << "Could not create capture instances...\n";
		exit(1);
	}

	cout<<"Opening video files\n";
	capture_left->open(filename_left);
	capture_right->open(filename_right);
	if((capture_left->isOpened())&&(capture_right->isOpened())){
		cout<<"Video files opened!\n";
	}else{
		cout<<"Video files could not be opened/not found...\n";
		exit(1);
	}

	width = capture_left->get(CV_CAP_PROP_FRAME_WIDTH);
	height = capture_left->get(CV_CAP_PROP_FRAME_HEIGHT);
}

module_file::~module_file(){

}

void module_file::getFrame(Mat* mat_left,Mat* mat_right){
	if((capture_left->isOpened())&&(capture_right->isOpened())){
		if(!capture_left->grab())EoF=true;
		if(!capture_right->grab())EoF=true;

		if(EoF==false){
			capture_left->retrieve(*mat_left,3);
			capture_right->retrieve(*mat_right,3);
			//cvtColor(*mat_left,*mat_left,CV_RGB2GRAY);
			//cvtColor(*mat_right,*mat_right,CV_RGB2GRAY);
		}
	}
}
