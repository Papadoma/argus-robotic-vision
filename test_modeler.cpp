#include "ogre_modeler.hpp"

int angles[3];

void make_sliders(){
	cv::createTrackbar("Yaw", "sliders", & angles[0],300);
	cv::createTrackbar("Pitch", "sliders", & angles[1],300);
	cv::createTrackbar("Roll", "sliders", & angles[2],300);
}



int main(){
	ogre_model model(800,600);

	cv::namedWindow("test");
	cv::namedWindow("sliders");

	cv::Point3f position = cv::Point3f(0,0,2000);
	cv::Point3f rot_vector = cv::Point3f(0,0,0);

	cv::Mat bones_rotation;

	angles[0]=150;
	angles[1]=150;
	angles[2]=150;

	make_sliders();
	int pos = 0;


	model.set_depth_limits(1000, 2000, 20000);
	model.set_camera_clip(1000, 20000);
	while(1)
	{
		bones_rotation = cv::Mat::zeros(19,3,CV_32FC1);

		int key = cv::waitKey(1) & 255;;
		if ( key == 27 ) break;	//Esc
		if ( key == 48 ) pos = 0;	//0
		if ( key == 49 ) pos = 1;	//1
		if ( key == 50 ) pos = 2;	//2
		if ( key == 51 ) pos = 3;	//3
		if ( key == 52 ) pos = 4;	//4
		if ( key == 53 ) pos = 5;	//5
		if ( key == 54 ) pos = 6;	//6
		if ( key == 55 ) pos = 7;	//7
		if ( key == 56 ) pos = 8;	//8
		if ( key == 57 ) pos = 9;	//9
		if ( key == 97 ) pos = 10;	//a
		if ( key == 98 ) pos = 11;	//b
		if ( key == 99 ) pos = 12;	//c
		if ( key == 100 ) pos = 13;	//d
		if ( key == 101 ) pos = 14;	//e
		if ( key == 102 ) pos = 15;	//f
		if ( key == 103 ) pos = 16;	//g
		if ( key == 104 ) pos = 17;	//h
		if ( key == 105 ) pos = 18;	//i
		if ( key == 32 ){
			angles[0]=150;
			angles[1]=150;
			angles[2]=150;
		}

		bones_rotation.at<float>(pos,0) = angles[0]-150;
		bones_rotation.at<float>(pos,1) = angles[1]-150;
		bones_rotation.at<float>(pos,2) = angles[2]-150;

		model.move_model(position, rot_vector, 700);
		model.rotate_bones(bones_rotation);

		double t = (double)cv::getTickCount();
		cv::Mat output = model.get_depth()->clone();
		t = (double)cv::getTickCount() - t;
		float fps = 1/(t/cv::getTickFrequency());
		std::cout << "[Modeler] Render fps " <<  model.get_fps() <<" Total fps "<< fps << std::endl;//for fps

		cv::cvtColor(output,output,CV_GRAY2RGB);

		std::ostringstream str;
		str << "Yaw:" << angles[0]-150 << " Pitch:" << angles[1]-150 << " Roll:" << angles[2]-150 << "  pos:" << pos;
		putText(output, str.str(), cv::Point(5,30), CV_FONT_HERSHEY_PLAIN, 2,CV_RGB(0,0,255));

		std::cout << model.get_2D_pos()<<std::endl;
		cv::circle(output,cv::Point(model.get_2D_pos().at<ushort>(0,0),model.get_2D_pos().at<ushort>(0,1)),2,cv::Scalar(0,0,255),-6);
		cv::circle(output,cv::Point(model.get_2D_pos().at<ushort>(1,0),model.get_2D_pos().at<ushort>(1,1)),2,cv::Scalar(0,0,255),-6);
		cv::circle(output,cv::Point(model.get_2D_pos().at<ushort>(2,0),model.get_2D_pos().at<ushort>(2,1)),2,cv::Scalar(0,0,255),-6);
		cv::circle(output,cv::Point(model.get_2D_pos().at<ushort>(3,0),model.get_2D_pos().at<ushort>(3,1)),2,cv::Scalar(0,0,255),-6);
		cv::circle(output,cv::Point(model.get_2D_pos().at<ushort>(4,0),model.get_2D_pos().at<ushort>(4,1)),2,cv::Scalar(0,0,255),-6);

		cv::Mat jet_depth_map2;
		cv::applyColorMap(output, jet_depth_map2, cv::COLORMAP_JET );
		//cv::imshow( "depth2", jet_depth_map2 );

		imshow("test",jet_depth_map2);
	}


	if(DEBUG_CONSOLE)std::cout<<"[Modeler] end of program"<<std::endl;
	return 1;
}

