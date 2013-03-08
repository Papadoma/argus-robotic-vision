#include "ogre_modeler.hpp"
#include <math.h>
#include <opencv2/opencv.hpp>


#define rand_num ((double) rand() / (RAND_MAX))

const float w = 0.5;
const float c1 = 1.5;
const float c2 = 1.5;

cv::Point input_head(140,120);
cv::Point input_hand_r(55,137);
cv::Point input_hand_l(224,122);
cv::Point input_foot_r(117,288);
cv::Point input_foot_l(166,282);
bool enable_bones = false;

const int swarm_size = 50;

struct particle_position{
	cv::Mat bones_rotation;
	cv::Point3f model_position;
	cv::Point3f model_rotation;
	float scale;
};

struct particle{
	int id;
	cv::Mat particle_silhouette;
	cv::Mat particle_depth;
	cv::Mat extremas;
	cv::Mat violation;

	particle_position current_position;
	particle_position next_position;
	particle_position best_position;
	float best_score;
};

class pose_estimator{
private:
	cv::RNG rand_gen;

	void load_param();
	void set_modeler_depth();
	cv::Mat calculate_depth_limit();

	cv::Point3f estimate_starting_position();
	void calc_next_position(particle&);
	float calc_score(particle&);
	void paint_particle(particle&);

	cv::Mat get_random_bones_rot(bool);
	cv::Point3f get_random_model_position(float, float);
	cv::Point3f get_random_model_rot();

	particle swarm[swarm_size];
	particle_position best_global_position;


	cv::Mat cameraMatrix, Q;
	cv::Mat bone_max, bone_min;

	cv::Point3f human_position;		//At first, the estimated human position;

	int startX, startY, endX, endY;		//2D search space for positioning the model
	int frame_width, frame_height;

	cv::Mat_<cv::Point3f> camera_viewspace; //TRN, TLN, BLN, BRN, TRF, TLF, BLF, BRF

	cv::Mat best_global_silhouette;
	cv::Mat best_global_depth;
public:
	float best_global_score;
	ogre_model* model;
	cv::Mat input_frame;
	cv::Mat input_silhouette;
	cv::Mat input_depth;

	void init_particles();
	void show_best_solution();
	pose_estimator(int, int);
	~pose_estimator();
	void calc_evolution();
	void randomize_bones();
};

pose_estimator::pose_estimator(int frame_width, int frame_height)
:frame_width(frame_width),
 frame_height(frame_height)
{
	std::cout << "[Pose Estimator] New Pose estimator: " << frame_width << "x" << frame_height << std::endl;
	load_param();

	input_frame = cv::Mat::zeros(frame_height, frame_width, CV_8UC1);
	cv::Mat input_frameROI = cv::imread("woman_dancing.png",0);
	//cv::Mat input_frameROI = cv::imread("man_standing.png",0);

	//Fix the input_frame size to the pose_estimator's size
	startX = (frame_width - input_frameROI.cols)/2;
	startY = (frame_height - input_frameROI.rows)/2;
	endX = startX + input_frameROI.cols;
	endY = startY + input_frameROI.rows;
	input_frameROI.copyTo(input_frame(cv::Rect(startX,startY,input_frameROI.cols,input_frameROI.rows)));

	input_frame.copyTo(input_depth);
	threshold(input_frame,input_silhouette,1,255,cv::THRESH_BINARY);


	input_head+= cv::Point(startX,startY);
	input_hand_r+=cv::Point(startX,startY);
	input_hand_l+=cv::Point(startX,startY);
	input_foot_r+=cv::Point(startX,startY);
	input_foot_l+=cv::Point(startX,startY);


	model = new ogre_model(frame_width,frame_height);
	set_modeler_depth();
	camera_viewspace = model->get_camera_viewspace();

	init_particles();
}



pose_estimator::~pose_estimator(){
	cv::destroyAllWindows();
	delete(model);
}

/**
 * Load camera parameters
 */
void pose_estimator::load_param(){
	std::cout << "[Pose Estimator] Loading settings"<< std::endl;
	std::string intrinsics="intrinsics_eye.yml";
	std::string extrinsics="extrinsics_eye.yml";
	std::string bone_limits="bone_limits.yml";

	cv::Size imageSize(frame_width,frame_height);

	cv::FileStorage fs(intrinsics, CV_STORAGE_READ);
	if( fs.isOpened() )
	{
		fs["M1"] >> cameraMatrix;
		fs.release();
	}
	else
		std::cout << "Error: can not load the intrinsic parameters" << std::endl;

	fs.open(extrinsics, CV_STORAGE_READ);
	if( fs.isOpened() )
	{
		fs["Q"] >> Q;
		fs.release();
	}
	else
		std::cout << "Error: can not load the extrinsics parameters" << std::endl;

	fs.open(bone_limits, CV_STORAGE_READ);
	if( fs.isOpened() )
	{
		fs["MinLimits"] >> bone_min;
		fs["MaxLimits"] >> bone_max;
		fs.release();
	}
	else
		std::cout << "Error: can not load the bone limits" << std::endl;
}

/**
 * Sets the modelers depth, so the disparity map of the modeler matches
 * the real disparity map. This means that the 2 world coordinate
 * systems match perfectly.
 */
inline void pose_estimator::set_modeler_depth(){
	std::cout << "[Pose Estimator] Setting model depth"<< std::endl;
	cv::Mat limits = calculate_depth_limit();

	model->set_depth_limits(limits.at<double>(2),limits.at<double>(1),limits.at<double>(0));
	model->set_camera_clip(limits.at<double>(2),limits.at<double>(0));
	std::cout<<"Depth set to "<<limits.at<double>(2)<<","<<limits.at<double>(1)<<","<<limits.at<double>(0)<<std::endl;
}

/**
 * Calculates and returns the min, max and center depth value calculated by the stereo rig
 * on the center of the left camera. This is done with respect to the real camera's world
 * coordinate system.
 */
cv::Mat pose_estimator::calculate_depth_limit(){
	std::cout << "[Pose Estimator] Calculating depth limits"<< std::endl;
	cv::Mat result;
	cv::Mat values = cv::Mat(3,1,CV_64FC1);

	cv::Mat disparity_limit = cv::Mat(4,3,CV_8UC1);
	disparity_limit.at<uchar>(0,0) = frame_width/2;
	disparity_limit.at<uchar>(0,1) = frame_width/2;
	disparity_limit.at<uchar>(0,2) = frame_width/2;
	disparity_limit.at<uchar>(1,0) = frame_height/2;
	disparity_limit.at<uchar>(1,1) = frame_height/2;
	disparity_limit.at<uchar>(1,2) = frame_height/2;
	disparity_limit.at<uchar>(2,0) = 1;
	disparity_limit.at<uchar>(2,1) = 16;
	disparity_limit.at<uchar>(2,2) = 32;
	disparity_limit.at<uchar>(3,0) = 1;
	disparity_limit.at<uchar>(3,1) = 1;
	disparity_limit.at<uchar>(3,2) = 1;
	disparity_limit.convertTo(disparity_limit,CV_64FC1);

	result = Q * disparity_limit;
	values.at<double>(0) = ceil(result.at<double>(2,0)/result.at<double>(3,0));
	values.at<double>(1) = round(result.at<double>(2,1)/result.at<double>(3,1));
	values.at<double>(2) = floor(result.at<double>(2,2)/result.at<double>(3,2));
	return values;
}


/**
 * Initializes swarm particles and the best position
 */
void pose_estimator::init_particles(){
	std::cout << "[Pose Estimator] Initializing particles"<< std::endl;
	srand(time(0));
	cv::theRNG().state = time(0);

	human_position = estimate_starting_position();
	std::cout<< "[Pose Estimator] estimated starting position" << human_position <<std::endl;

	for(int i = 0; i<swarm_size; i++){
		swarm[i].id = i;
		//swarm[i].current_position.bones_rotation = get_random_bones_rot(true);
		swarm[i].current_position.bones_rotation = cv::Mat::zeros(19,3,CV_32FC1);
		swarm[i].current_position.model_position = cv::Point3f(0,0,human_position.z);
		swarm[i].current_position.model_rotation = get_random_model_rot();
		swarm[i].current_position.scale = round(100 + rand_num*(1000));
		//swarm[i].current_position.scale = 700;

		swarm[i].next_position.bones_rotation = get_random_bones_rot(false);
		swarm[i].next_position.model_position = get_random_model_position(human_position.z*0.9,human_position.z*1.1);
		swarm[i].next_position.model_rotation = get_random_model_rot();
		swarm[i].next_position.scale = round(100 + rand_num*(1000));
		//swarm[i].next_position.scale = 700;

		swarm[i].best_position.bones_rotation = cv::Mat::zeros(19,3,CV_32FC1);
		swarm[i].best_position.model_position = cv::Point3f(0,0,human_position.z);
		swarm[i].best_position.model_rotation =  cv::Point3f(0,0,0);
		swarm[i].best_position.scale = round(100 + rand_num*(1000));
		//swarm[i].best_position.scale = 700;

		swarm[i].best_score = 0;
		swarm[i].violation = cv::Mat::ones(19,3,CV_32FC1);
	}
	best_global_position.bones_rotation = cv::Mat::zeros(19,3,CV_32FC1);
	best_global_position.model_position = cv::Point3f(0,0,human_position.z);
	best_global_position.model_rotation = cv::Point3f(0,0,0);
	best_global_position.scale = 700;

	best_global_score = 0;
}

/**
 * Estimate where the human is relative to the camera
 */
cv::Point3f pose_estimator::estimate_starting_position(){
	cv::Mat result, disparity;
	input_frame.convertTo(disparity, CV_32FC1,32./255);
	cv::reprojectImageTo3D(disparity, result, Q);
	cv::Scalar mean_depth = mean(result, input_silhouette);
	return cv::Point3f(mean_depth.val[0], mean_depth.val[1], mean_depth.val[2]);
}

/**
 * Calculates random values for the bones based on loaded movement
 * restrictions.
 */
inline cv::Mat pose_estimator::get_random_bones_rot(bool initial){
	if(enable_bones){
		cv::Mat result = cv::Mat::ones(19,3,CV_32FC1);
		for(int i=0 ; i<19 ; i++){
			for(int j=0 ; j<3 ; j++){
				float random_number =  rand_gen.gaussian(0.3);
				if(initial){
					random_number < 0 ? random_number = 0 : random_number;
					random_number > 1 ? random_number = 1 : random_number;
				}
				result.at<float>(i,j) = random_number * result.at<float>(i,j);
				//std::cout<< random_number <<std::endl;
				//result.at<float>(i,j) = rand_num * result.at<float>(i,j);		//uniform distribution
			}
		}
		if(initial){
			result = bone_max.mul(result) + bone_min.mul((1-result));
		}else{
			result = result*1;
		}
		return result;
	}else{
		return cv::Mat::zeros(19,3,CV_32FC1);
	}
}

/**
 * Calculates random values for model position
 */
inline cv::Point3f pose_estimator::get_random_model_position(float min_z = 0 ,float max_z = 0){

	if(min_z == 0 || max_z == 0){
		min_z=camera_viewspace(0,0).z;
		max_z=camera_viewspace(4,0).z;
	}

	cv::Point3f position;
	position.z = min_z+500 + rand_num * (max_z - min_z);
	//std::cout<<camera_viewspace<<std::endl;
	//std::cout<<"random Z "<<position.z<<std::endl;
	//0:TRN, 1:TLN, 2:BLN, 3:BRN, 4:TRF, 5:TLF, 6:BLF, 7:BRF

	float min_x = (camera_viewspace(4,0).x - camera_viewspace(0,0).x)/(max_z-min_z)*position.z + (camera_viewspace(0,0).x*max_z - min_z*camera_viewspace(4,0).x)/(max_z-min_z);
	float max_x = (camera_viewspace(5,0).x - camera_viewspace(1,0).x)/(max_z-min_z)*position.z + (camera_viewspace(1,0).x*max_z - min_z*camera_viewspace(5,0).x)/(max_z-min_z);
	float min_y = (camera_viewspace(6,0).y - camera_viewspace(2,0).y)/(max_z-min_z)*position.z + (camera_viewspace(2,0).y*max_z - min_z*camera_viewspace(6,0).y)/(max_z-min_z);
	float max_y = (camera_viewspace(5,0).y - camera_viewspace(1,0).y)/(max_z-min_z)*position.z + (camera_viewspace(1,0).y*max_z - min_z*camera_viewspace(5,0).y)/(max_z-min_z);
	//std::cout << " minX " << min_x << " max_x "<< max_x << " min_y " << min_y<< " max_y "<< max_y << std::endl;

	//position.x = min_x + rand_num * (max_x - min_x);	//uniform distribution
	//position.y = min_y + rand_num * (max_y - min_y);
	position.x = rand_gen.gaussian(0.3) * 0.5 * (max_x - min_x);	//uniform distribution
	position.y = rand_gen.gaussian(0.3) * 0.5 * (max_y - min_y);

	return position;
}

/**
 * Calculates random values for model global rotation
 */
inline cv::Point3f pose_estimator::get_random_model_rot(){
	//	float yaw = 0 + rand_num*(180);
	//	float pitch = 0 + rand_num*(360);
	//	float roll = 0 + rand_num*(360);

	float yaw = rand_gen.gaussian(0.3)*(180);
	float pitch = rand_gen.gaussian(0.3)*(180);
	float roll = rand_gen.gaussian(0.3)*(180);


	return cv::Point3f(yaw, pitch, roll);
}

/**
 *Refreshes the scores and thus evolving the swarm
 */
void pose_estimator::calc_evolution(){
	for(int i=0;i<swarm_size;i++){
		paint_particle(swarm[i]);
	}
	for(int i=0;i<swarm_size;i++){
		float score = calc_score(swarm[i]);
		if(score > swarm[i].best_score){
			swarm[i].best_score = score;
			swarm[i].best_position = swarm[i].current_position;
		}
		if(swarm[i].best_score > best_global_score){
			best_global_score = swarm[i].best_score;
			best_global_position = swarm[i].best_position;
		}
	}
	for(int i=0;i<swarm_size;i++){
		calc_next_position(swarm[i]);
	}
}

inline void pose_estimator::paint_particle(particle& particle_inst){
	model->move_model(particle_inst.current_position.model_position,particle_inst.current_position.model_rotation, particle_inst.current_position.scale);
	model->rotate_bones(particle_inst.current_position.bones_rotation);
	particle_inst.extremas = model->get_2D_pos();
	particle_inst.particle_depth = model->get_depth()->clone();
	threshold(particle_inst.particle_depth,particle_inst.particle_silhouette,1,255,cv::THRESH_BINARY);

}

/**
 *Calculates the best next position for the given particle, based on
 *random movement, its best position and the global best one
 */
void pose_estimator::calc_next_position(particle& particle_inst){
	if(!enable_bones)particle_inst.next_position.model_position = w*particle_inst.next_position.model_position + c1*rand_num*(particle_inst.best_position.model_position-particle_inst.current_position.model_position) + c2*rand_num*(best_global_position.model_position-particle_inst.current_position.model_position);
	particle_inst.next_position.model_rotation = w*particle_inst.next_position.model_rotation + c1*rand_num*(particle_inst.best_position.model_rotation-particle_inst.current_position.model_rotation) + c2*rand_num*(best_global_position.model_rotation-particle_inst.current_position.model_rotation);
	if(enable_bones)particle_inst.next_position.bones_rotation = w*particle_inst.next_position.bones_rotation.mul(particle_inst.violation) + c1*rand_num*(particle_inst.best_position.bones_rotation-particle_inst.current_position.bones_rotation) + c2*rand_num*(best_global_position.bones_rotation-particle_inst.current_position.bones_rotation);
	if(!enable_bones)particle_inst.next_position.scale = w*particle_inst.next_position.scale + c1*rand_num*(particle_inst.best_position.scale-particle_inst.current_position.scale) + c2*rand_num*(best_global_position.scale-particle_inst.current_position.scale);

	if(!enable_bones)particle_inst.current_position.model_position += particle_inst.next_position.model_position;
	particle_inst.current_position.model_rotation += particle_inst.next_position.model_rotation;
	if(enable_bones)particle_inst.current_position.bones_rotation += particle_inst.next_position.bones_rotation;
	if(!enable_bones)particle_inst.current_position.scale += particle_inst.next_position.scale;


	//Place the new bones rotation within limits
	cv::Mat result = particle_inst.current_position.bones_rotation.clone();
	cv::min(particle_inst.current_position.bones_rotation,bone_max,particle_inst.current_position.bones_rotation);
	cv::max(particle_inst.current_position.bones_rotation,bone_min,particle_inst.current_position.bones_rotation);
	particle_inst.violation = result == particle_inst.current_position.bones_rotation;	//Check if before and after limiting
	particle_inst.violation.convertTo(particle_inst.violation, CV_32FC1,(float)1/255);
	if (57-cv::countNonZero(particle_inst.violation)){
		std::cout << "[Pose Estimator]: Violation fixed on particle: "<<particle_inst.id<<" for No of values: "<< 57-cv::countNonZero(particle_inst.violation)<<std::endl;
	}
}

/**
 *Calculates fitness score
 */
float pose_estimator::calc_score(particle& particle_inst){
	cv::Mat result_xor,result_and;
	//bitwise_xor(input_frame, particle_inst.particle_silhouette, result_xor);
	bitwise_and(input_silhouette, particle_inst.particle_silhouette, result_and);

	cv::Mat result;
	cv::absdiff(particle_inst.particle_depth,input_depth,result);

	if(countNonZero(result_and)){
		//cv::bitwise_and(result_and,result,result);
		//std::cout<< "[Pose Estimator]: percentage of difference between input and estimation depth " <<1./(1+mean(result).val[0])<<std::endl;
		return 0.5 * 1./(1+mean(result).val[0]) + 0.5 * countNonZero(result_and)/fmaxf(countNonZero(input_silhouette),countNonZero(particle_inst.particle_silhouette));
		//return 1./(1+mean(result).val[0]);
	}else{
		return 0;
	}

	//imshow("123",result);

	//	particle_inst.particle_depth.convertTo(temp,CV_16UC1,16*32/255);
	//	cv::absdiff(temp,input_depth,result);



	float dist1 = 1./(1 + sqrt(pow((particle_inst.extremas.at<ushort>(0,0)-input_head.x),2) + pow((particle_inst.extremas.at<ushort>(0,1)-input_head.y),2)));
	float dist2 = 1./(1 + sqrt(pow((particle_inst.extremas.at<ushort>(1,0)-input_hand_r.x),2) + pow((particle_inst.extremas.at<ushort>(1,1)-input_hand_r.y),2)));
	float dist3 = 1./(1 + sqrt(pow((particle_inst.extremas.at<ushort>(2,0)-input_hand_l.x),2) + pow((particle_inst.extremas.at<ushort>(2,1)-input_hand_l.y),2)));
	float dist4 = 1./(1 + sqrt(pow((particle_inst.extremas.at<ushort>(3,0)-input_foot_r.x),2) + pow((particle_inst.extremas.at<ushort>(3,1)-input_foot_r.y),2)));
	float dist5 = 1./(1 + sqrt(pow((particle_inst.extremas.at<ushort>(4,0)-input_foot_l.x),2) + pow((particle_inst.extremas.at<ushort>(4,1)-input_foot_l.y),2)));
	//std::cout<< "Distances! "<<dist1<<" "<<dist2<<" "<<dist3<<" "<<dist4<<" "<<dist5<<" "<<std::endl;
	//return (dist1 + dist2 + dist3 + dist4 + dist5)/5 * countNonZero(result_and)/fmaxf(countNonZero(input_silhouette),countNonZero(particle_inst.particle_silhouette));
	//return countNonZero(result_and)/fmaxf(countNonZero(input_silhouette),countNonZero(particle_inst.particle_silhouette));
}

void pose_estimator::show_best_solution(){
	model->move_model(best_global_position.model_position,best_global_position.model_rotation, best_global_position.scale);
	model->rotate_bones(best_global_position.bones_rotation);
	best_global_depth = model->get_depth()->clone();



	threshold(best_global_depth,best_global_silhouette,1,255,cv::THRESH_BINARY);
	cv::Mat result, resullt2 = cv::Mat::zeros(640,480,CV_8UC1);
	bitwise_xor(input_silhouette, best_global_silhouette, result);
	imshow("best global silhouette", best_global_silhouette);
	cv::Mat jet_depth_map2;
	cv::applyColorMap(best_global_depth, jet_depth_map2, cv::COLORMAP_JET );

	cv::Mat extremas = model->get_2D_pos();
	for(int i=0;i<5;i++){
		cv::circle(jet_depth_map2,cv::Point(extremas.at<ushort>(i,0),extremas.at<ushort>(i,1)),3,cv::Scalar(0,255,255),2);
		cv::circle(jet_depth_map2,cv::Point(extremas.at<ushort>(i,0),extremas.at<ushort>(i,1)),1,cv::Scalar(0,0,255),2);
	}

	imshow("best global depth", jet_depth_map2);
	imshow("best global diff", result);


	//std::cout<< cv::countNonZero(resullt2)<< std::endl;
	//imshow("123",resullt2);

	std::cout<< "[Pose Estimator] best global score: " <<best_global_score << " best global position: " << best_global_position.model_position <<" best global scale: "<<best_global_position.scale <<std::endl;

}

void pose_estimator::randomize_bones(){
	for(int i = 0; i<swarm_size; i++){
		swarm[i].current_position.bones_rotation = get_random_bones_rot(true);
		swarm[i].next_position.bones_rotation = get_random_bones_rot(false);
		//swarm[i].best_position.bones_rotation =  cv::Mat::zeros(19,3,CV_32FC1);
		//swarm[i].best_score=0;
	}
	//best_global_score = 0;
}

int main(){
	pose_estimator instance(640,480);
	cv::namedWindow("input frame");
	cv::namedWindow("best global silhouette");
	cv::namedWindow("best global depth");
	cv::namedWindow("best global diff");

	imshow("input frame", instance.input_frame);



	while(1)
	{
		int key = cv::waitKey(1) & 255;;
		if ( key == 27 ) break;	//Esc
		if ( key == 114 ) { //R
			instance.init_particles();
		}
		if ( key == 116 ){ //T
			enable_bones = !enable_bones;
			instance.randomize_bones();
		}

		cv::Mat test = instance.input_frame.clone();
		cv::cvtColor(test,test,CV_GRAY2RGB);
		cv::circle(test,input_head,3,cv::Scalar(0,0,255),2);
		cv::circle(test,input_hand_r,3,cv::Scalar(0,0,255),2);
		cv::circle(test,input_hand_l,3,cv::Scalar(0,0,255),2);
		cv::circle(test,input_foot_r,3,cv::Scalar(0,0,255),2);
		cv::circle(test,input_foot_l,3,cv::Scalar(0,0,255),2);
		imshow("input frame", test);

		instance.show_best_solution();

		instance.calc_evolution();
	}
}

