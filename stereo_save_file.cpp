#include <stdio.h>
#include <opencv.hpp>

#include "module_input.hpp"

using namespace std;
using namespace cv;

class stereo_save_file{
private:
	module_eye input_module;

	VideoWriter video_left;
	VideoWriter video_right;

	Mat mat_left;
	Mat mat_right;

	Rect roi1, roi2;
	Mat rmap[2][2];

	int width;
	int height;

	void load_param();

public:
	bool flag_recording;

	stereo_save_file();
	~stereo_save_file();

	void refresh_frame();
	void refresh_window();
	void saveFrame();

	void info();
};

//Constructor
stereo_save_file::stereo_save_file(){
	Size framesize=input_module.getSize();
	width = framesize.width;
	height = framesize.height;

	video_left.open("left.mpg",CV_FOURCC('P','I','M','1'),40,framesize,true);
	video_right.open("right.mpg",CV_FOURCC('P','I','M','1'),40,framesize,true);

	flag_recording=false;

	if( !video_left.isOpened() || !video_right.isOpened() ) {
		cout<<"Could not create files\n";
		exit(1);
	}

}

//Destructor
stereo_save_file::~stereo_save_file(){
	destroyAllWindows();
}

void stereo_save_file::refresh_frame(){
	input_module.getFrame(mat_left ,mat_right);
}

void stereo_save_file::refresh_window(){

	Mat imgResult(height,2*width,CV_8UC3); // Your final image
	Mat roiImgResult_Left = imgResult(Rect(0,0,width,height));
	Mat roiImgResult_Right = imgResult(Rect(width,0,width,height));
	Mat roiImg1 = mat_left(Rect(0,0,width,height));
	Mat roiImg2 = mat_right(Rect(0,0,width,height));
	roiImg1.copyTo(roiImgResult_Left);
	roiImg2.copyTo(roiImgResult_Right);


	if(flag_recording){
		putText(imgResult, "RECORDING", Point(10,30), FONT_HERSHEY_SIMPLEX, 1, Scalar (0,0,255),2);
	}else{
		putText(imgResult, "Ready...", Point(10,30), FONT_HERSHEY_SIMPLEX, 1, Scalar (0,255,0),2);
	}
	imshow( "original_camera", imgResult );

}

void stereo_save_file::saveFrame(){
	if(flag_recording){
		video_left<<mat_left;
		video_right<<mat_right;
	}
}

int main(){

	int key_pressed=255;
	stereo_save_file *eye_stereo = new stereo_save_file();

	while(1){

		eye_stereo->refresh_frame();
		eye_stereo->saveFrame();
		eye_stereo->refresh_window();
		key_pressed = cvWaitKey(1) & 255;

		if ( key_pressed == 27 ) break;
		if ( key_pressed == 32 ) eye_stereo->flag_recording = (!eye_stereo->flag_recording);

	}

	delete(eye_stereo);
	return 0;
}
