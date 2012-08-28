#include <stdio.h>
#include <opencv.hpp>

#include "module_eye.hpp"
#include "module_file.hpp"

using namespace std;
using namespace cv;

class stereo_save_file{
private:

	Mat* mat_left;
	Mat* mat_right;

	Rect roi1, roi2;
	Mat rmap[2][2];

	int width;
	int height;

	void load_param();

public:
	stereo_save_file();
	~stereo_save_file();

	Mat imHist(Mat, float, float);


	void refresh_frame();
	void refresh_window();
	void compute_depth();

	void info();
};

//Constructor
stereo_save_file::stereo_save_file(){
	width = 640;
	height = 480;


	mat_left=new Mat(height,width,CV_8UC3);
	mat_right=new Mat(height,width,CV_8UC3);


}

//Destructor
stereo_save_file::~stereo_save_file(){
	destroyAllWindows();
}

void stereo_save_file::refresh_frame(){

}

void stereo_save_file::refresh_window(){
	imshow( "original_camera_left", *mat_left );
	imshow( "original_camera_right", *mat_right );
}

int main(){

	int key_pressed=255;
	stereo_save_file *eye_stereo = new stereo_save_file();

	while(1){

		eye_stereo->refresh_frame();

		eye_stereo->refresh_window();
		key_pressed = cvWaitKey(1) & 255;
		if ( key_pressed == 27 ) break;

	}

	delete(eye_stereo);
	return 0;
}
