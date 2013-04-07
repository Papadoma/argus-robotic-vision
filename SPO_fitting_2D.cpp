#include <opencv2/opencv.hpp>
#include <math.h>
#include <time.h>

#define rand_num ((double) rand() / (RAND_MAX))

const float length = 120;
const int swarm_size = 10;

const float w = 0.5;
const float c1 = 1.5;
const float c2 = 1.5;

struct particle_position{
	cv::Point particle_center;
	float particle_rot;
	float joint_angle;
	float scale;
};

struct particle{
	cv::Mat particle_silhouette;
	particle_position current_position;
	particle_position next_position;

	particle_position best_position;
	float best_score;
};

//[particle_center, particle_rot, joint_angle,scale]

class SPO_fitting{
private:
	cv::Mat source_image;

	void load_source();
	void paint_particle(particle&);
	float calc_score(particle&);
	void calc_next_position(particle&);
	void init_particles();

	particle swarm[swarm_size];

	particle_position best_global_position;
	float best_global_score;

public:
	SPO_fitting(std::string);
	//~SPO_fitting();
	void show();
	void calc_evolution();
};

SPO_fitting::SPO_fitting(std::string filename){
	source_image = cv::imread(filename,0);
	best_global_score = 0;
	init_particles();
}

void SPO_fitting::init_particles(){
	cv::RNG rng;
	srand(time(0));

	for(int i = 0; i<swarm_size; i++){
		swarm[i].particle_silhouette = cv::Mat::zeros(source_image.size(),CV_8UC1);

		swarm[i].current_position.particle_center.x = 0 +  rand_num * (source_image.cols-0);
		swarm[i].current_position.particle_center.y = 0 +  rand_num * (source_image.rows-0);
		swarm[i].current_position.particle_rot = 0 +  rand_num * (360-0);
		swarm[i].current_position.joint_angle = 0 +  rand_num * (360-0);
		swarm[i].current_position.scale = 0 +  rand_num * (1-0);

		swarm[i].next_position.particle_center.x = 0 +  rand_num * (source_image.cols-0);
		swarm[i].next_position.particle_center.y = 0 +  rand_num * (source_image.rows-0);
		swarm[i].next_position.particle_rot = 0 +  rand_num * (360-0);
		swarm[i].next_position.joint_angle = 0 +  rand_num * (360-0);
		swarm[i].next_position.scale = 0 +  rand_num * (2-0);

		std::cout <<swarm[i].next_position.scale <<std::endl;
		paint_particle(swarm[i]);
	}
	best_global_position.particle_center.x = 0 +  rand_num * (source_image.cols-0);
	best_global_position.particle_center.y = 0 +  rand_num * (source_image.rows-0);
	best_global_position.particle_rot = 0 +  rand_num * (360-0);
	best_global_position.joint_angle = 0 +  rand_num * (360-0);
	best_global_position.scale = 0 +  rand_num * (1-0);
}

inline void SPO_fitting::paint_particle(particle& particle_inst){
	particle_inst.particle_silhouette = cv::Mat::zeros(source_image.size(),CV_8UC1);

	cv::Point start;
	cv::Point middle;
	cv::Point end;
	start.x = particle_inst.current_position.particle_center.x;
	start.y = particle_inst.current_position.particle_center.y;
	middle.x = particle_inst.current_position.particle_center.x + length * cos(particle_inst.current_position.particle_rot*M_PI/180);
	middle.y = particle_inst.current_position.particle_center.y + length * sin(particle_inst.current_position.particle_rot*M_PI/180);
	//if(middle.x<0)middle.x=0;
	//if(middle.y<0)middle.y=0;
	end.x = middle.x + length * cos(particle_inst.current_position.joint_angle*M_PI/180);
	end.y = middle.y + length * sin(particle_inst.current_position.joint_angle*M_PI/180);
	//if(end.x<0)end.x=0;
	//if(end.y<0)end.y=0;
	//std::cout << start<< middle<<end <<std::endl;

	line(particle_inst.particle_silhouette, start, middle, cv::Scalar(255), 20);
	line(particle_inst.particle_silhouette, middle, end, cv::Scalar(255), 20);
	//std::cout << start << end << std::endl;
}

inline void SPO_fitting::calc_evolution(){
	float score;
	std::cout<<best_global_score<<std::endl;

	for(int i=0;i<swarm_size;i++){
		paint_particle(swarm[i]);
		score = calc_score(swarm[i]);
		if(score > swarm[i].best_score){
			swarm[i].best_score = score;
			swarm[i].best_position = swarm[i].current_position;
		}
		if(score > best_global_score){
			best_global_score = score;
			best_global_position = swarm[i].current_position;
		}
	}
	for(int i=0;i<swarm_size;i++){
		calc_next_position(swarm[i]);
	}
}

inline void SPO_fitting::calc_next_position(particle& particle_inst){
	particle_inst.next_position.particle_center = w*particle_inst.next_position.particle_center + c1*rand_num*(particle_inst.best_position.particle_center-particle_inst.current_position.particle_center) + c2*rand_num*(best_global_position.particle_center-particle_inst.current_position.particle_center);
	particle_inst.next_position.particle_rot = w*particle_inst.next_position.particle_rot + c1*rand_num*(particle_inst.best_position.particle_rot-particle_inst.current_position.particle_rot) + c2*rand_num*(best_global_position.particle_rot-particle_inst.current_position.particle_rot);
	particle_inst.next_position.joint_angle = w*particle_inst.next_position.joint_angle + c1*rand_num*(particle_inst.best_position.joint_angle-particle_inst.current_position.joint_angle) + c2*rand_num*(best_global_position.joint_angle-particle_inst.current_position.joint_angle);
	particle_inst.next_position.scale = w*particle_inst.next_position.scale + c1*rand_num*(particle_inst.best_position.scale-particle_inst.current_position.scale) + c2*rand_num*(best_global_position.scale-particle_inst.current_position.scale);

	particle_inst.current_position.particle_center += particle_inst.next_position.particle_center;
	particle_inst.current_position.particle_rot += particle_inst.next_position.particle_rot;
	particle_inst.current_position.joint_angle += particle_inst.next_position.joint_angle;
	particle_inst.current_position.scale += particle_inst.next_position.scale;

}

inline float SPO_fitting::calc_score(particle& particle_inst){
	cv::Mat result;
	bitwise_and(source_image, particle_inst.particle_silhouette, result);
	//imshow("test",result);
	//cv::waitKey(500);

	return countNonZero(result);

}

inline void SPO_fitting::show(){
	cv::Mat result_buffer;
	//source_image.copyTo(result_buffer);
	result_buffer = cv::Mat::zeros(source_image.size(),CV_8UC1);

	for(int i=0;i<swarm_size;i++){
		//addWeighted(result_buffer, 1-(double)1/swarm_size , swarm[i].particle_silhouette, (double)1/swarm_size, 0, result_buffer);
		result_buffer+=swarm[i].particle_silhouette;
	}
	//addWeighted(result_buffer, 0.5 , source_image, 0.5, 0, result_buffer);
	bitwise_xor(result_buffer,source_image,result_buffer);
	imshow("source", source_image);
	imshow("result_buffer", result_buffer);

	particle best_candidate;
	best_candidate.current_position = best_global_position;
	paint_particle(best_candidate);
	absdiff(best_candidate.particle_silhouette,source_image,best_candidate.particle_silhouette);
	imshow("best_candidate", best_candidate.particle_silhouette);

	//	imshow("particle 1",swarm[0].particle_silhouette);
	//	imshow("particle 2",swarm[1].particle_silhouette);
	//	imshow("particle 3",swarm[2].particle_silhouette);
}

int main(){
	SPO_fitting fitting_procedure("source_silhouette2.png");
	fitting_procedure.show();
	cvWaitKey(10);
	while(1){
		fitting_procedure.show();
		fitting_procedure.calc_evolution();

		int key_pressed = cvWaitKey(0) & 255;
		if ( key_pressed == 27 ) break;
	}

	return 0;
}
