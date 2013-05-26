#include "ogre_modeler.hpp"

int model_position[3];
int model_rotation[3];
int bone_rotation[3];
int scale;
int pose_num = 0;

void make_sliders(){
	cv::createTrackbar("X","position_sliders", & model_position[0],2000);
	cv::createTrackbar("Y","position_sliders", & model_position[1],2000);
	cv::createTrackbar("Z","position_sliders", & model_position[2],3000);
	cv::createTrackbar("scale","position_sliders", & scale,1400);
	cv::createTrackbar("yaw","position_sliders", & model_rotation[0],360);
	cv::createTrackbar("pitch","position_sliders", & model_rotation[1],360);
	cv::createTrackbar("roll","position_sliders", & model_rotation[2],360);

	cv::setTrackbarPos("X","position_sliders", 1000);
	cv::setTrackbarPos("Y","position_sliders", 1000);
	cv::setTrackbarPos("Z","position_sliders", 2000);
	cv::setTrackbarPos("scale","position_sliders", 600);
	cv::setTrackbarPos("yaw","position_sliders", 180);
	cv::setTrackbarPos("pitch","position_sliders", 180);
	cv::setTrackbarPos("roll","position_sliders", 180);


	cv::createTrackbar("bone yaw", "bones_sliders", &bone_rotation[0],360);
	cv::createTrackbar("bone pitch", "bones_sliders", &bone_rotation[1],360);
	cv::createTrackbar("bone roll", "bones_sliders", &bone_rotation[2],360);
	cv::setTrackbarPos("bone yaw", "bones_sliders", 180);
	cv::setTrackbarPos("bone pitch", "bones_sliders", 180);
	cv::setTrackbarPos("bone roll", "bones_sliders", 180);


}

void calculate_result(cv::Mat input, cv::Mat estimation, std::vector<double>& result, std::vector<std::string>& desc){
	cv::Mat temp, temp2;
	input.convertTo(input,CV_32FC1,1./255);
	estimation.convertTo(estimation,CV_32FC1,1./255);
	//************absdiff*********
	cv::absdiff(input*255,estimation*255,temp);
	result[0] = 1./(cv::mean(temp).val[0]);
	desc[0]="1/mean(absdiff):";

	//**********bitwise_xor: mean***********
	cv::bitwise_xor(input*255, estimation*255,temp);
	result[1] = 1./(cv::mean(temp).val[0]);
	desc[1] = "1/mean(bitwise_xor):";

	//************bitwise and/or per element: mean*********
	cv::bitwise_or(input, estimation,temp);
	cv::bitwise_and(input, estimation,temp2);
	cv::divide(temp2,temp,temp);
	result[2] = cv::mean(temp).val[0];
	desc[2] = "mean(AND/OR):";

	//************bitwise and/or: mean*********
	cv::bitwise_or(input, estimation,temp);
	cv::bitwise_and(input, estimation,temp2);
	result[3] = cv::mean(temp2).val[0]/cv::mean(temp).val[0];
	desc[3] = "mean(AND)/mean(OR):";

	//************fuzzy and/or*********
	cv::max(input,estimation,temp);	//or
	cv::min(input,estimation,temp2); //and
	result[4] = cv::sum(temp2).val[0]/cv::sum(temp).val[0];
	desc[4] = "sum(fAND)/sum(fOR):";

	//************fuzzy xor*********
	cv::min(input,1-estimation,temp);
	cv::min(1-input,estimation,temp2);
	cv::max(temp,temp2,temp);
	result[5] = 1./(cv::sum(temp).val[0]);
	desc[5] = "1/sum(fXOR):";
}

void show_results(std::vector<double>& result, std::vector<std::string>& desc){
	cv::Mat table = cv::Mat::zeros(600,400,CV_8UC3);
	for(int i=0;i<(int)result.size();i++){
		std::stringstream ss;
		ss<<desc[i]<<result[i];
		cv::putText(table, ss.str(), cv::Point(5,i*30+30),0,0.5,cv::Scalar(0,255,0));
	}
	imshow("results",table);
}

void save_pose(cv::Mat estimation,std::vector<double>& result, std::vector<std::string>& desc, cv::Point3f position,  cv::Point3f rotation, float scale,cv::Mat mat_bones_rotation){
	std::stringstream ss;

	ss<<"pose"<<pose_num<<".jpg";
	cv::applyColorMap(estimation, estimation, cv::COLORMAP_JET );
	imwrite(ss.str(), estimation );

	ss.str( std::string() );
	ss.clear();
	ss<<"pose"<<pose_num<<".yml";
	cv::FileStorage fs(ss.str(), cv::FileStorage::WRITE);

	fs<<"Model position"<<position;
	fs<<"Model rotation"<<rotation;
	fs<<"Model scale"<<scale;
	fs<<"Bones rotation"<<mat_bones_rotation;

	for(int i=0;i<(int)result.size();i++){
		if(desc[i]!=""){
			fs<<"Desc"<<desc[i]<<"Score"<<result[i];
		}
	}
	fs.release();
	pose_num++;
}

int main(){
	ogre_model model(640,480);

	cv::namedWindow("test");
	cv::namedWindow("position_sliders",CV_WINDOW_NORMAL );
	cv::namedWindow("bones_sliders",CV_WINDOW_NORMAL );

	cv::Point3f position;
	cv::Point3f rotation;
	cv::Mat mat_bones_rotation = cv::Mat::zeros(19,3,CV_32FC1);

	make_sliders();

	model.set_depth_limits(1000, 2000, 20000);
	model.set_camera_clip(1000, 20000);
	int pos = 0;
	int prev = 0;
	std::deque<double> mean_time;

	cv::Mat input = cv::imread("snap_depth3.png",0);

	while(1)
	{
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

		if (prev!=pos){
			cv::setTrackbarPos("bone yaw", "bones_sliders", mat_bones_rotation.at<float>(pos,0)+180);
			cv::setTrackbarPos("bone pitch", "bones_sliders", mat_bones_rotation.at<float>(pos,1)+180);
			cv::setTrackbarPos("bone roll", "bones_sliders", mat_bones_rotation.at<float>(pos,2)+180);
		}
		prev = pos;
		mat_bones_rotation.at<float>(pos,0)=bone_rotation[0]-180;
		mat_bones_rotation.at<float>(pos,1)=bone_rotation[1]-180;
		mat_bones_rotation.at<float>(pos,2)=bone_rotation[2]-180;

		position = cv::Point3f(model_position[0]-1000,model_position[1]-1000,model_position[2]);
		rotation = cv::Point3f(model_rotation[0]-180,model_rotation[1]-180,model_rotation[2]-180);

		model.move_model(position, rotation, scale);
		model.rotate_bones(mat_bones_rotation);
		double t = (double)cv::getTickCount();
		cv::Mat output = model.get_depth()->clone();
		t = (double)cv::getTickCount() - t;
		t = t*1000./cv::getTickFrequency();

		mean_time.push_back(t);
		if((int)mean_time.size()>240)mean_time.pop_front();
		double mean_value=0;
		for(int i=0;i<(int)mean_time.size();i++) mean_value += mean_time[i];

		std::cout << "[Modeler] Render fps " <<  model.get_fps() <<" Total ms "<< mean_value/(int)mean_time.size() << std::endl;//for fps

		output = 255-output;

		//*******************************
		std::vector<double> result(10);
		std::vector<std::string> text(10);
		calculate_result(input, output, result,text);
		show_results(result,text);
		if ( key == 's' )save_pose( output,result, text, position, rotation, scale, mat_bones_rotation);

		cv::cvtColor(output,output,CV_GRAY2BGR);

		std::ostringstream str;
		str << "Yaw:" << bone_rotation[0]-180 << " Pitch:" << bone_rotation[1]-180 << " Roll:" << bone_rotation[2]-180 << "  pos:" << pos;
		putText(output, str.str(), cv::Point(5,30), CV_FONT_HERSHEY_PLAIN, 2,CV_RGB(0,0,255));

		cv::Mat jet_depth_map2;
		cv::applyColorMap(output, jet_depth_map2, cv::COLORMAP_JET );
		//cv::imshow( "depth2", jet_depth_map2 );

		imshow("estimation",jet_depth_map2);
		imshow("input",input);
		cv::cvtColor(output,output,CV_BGR2GRAY);
		cv::Mat temp;
		cv::absdiff(input,output,temp);
		imshow("absdiff",temp*2);

	}


	if(DEBUG_CONSOLE)std::cout<<"[Modeler] end of program"<<std::endl;
	return 1;
}

