#include "pose_estimation.hpp"

float H_pose[]={
		0,	0,	0,		//head
		0,	0,	0,		//u_torso
		0,	0,	0,		//l_torso
		0,	0,	0,		//r_shoulder
		0,	0,	0,		//l_shoulder
		0,	0,	0,		//r_hip
		0,	0,	0,		//l_hip
		90,	0,	0,		//r_arm
		0,	0,	90,		//r_forearm
		0,	0,	0,		//r_hand
		-90,0,	0,		//l_arm
		0,	0,	-90,	//l_forearm
		0,	0,	0,		//l_hand
		0,	0,	0,		//r_thigh
		0,	0,	0,		//r_calf
		0,	0,	0,		//r_foot
		0,	0,	0,		//l_thigh
		0,	0,	0,		//l_calf
		0,	0,	0		//l_foot
};

float bones_penalty[]={
		0.5,	0.5,	0.5,	//head
		0.5,	0.5,	0.5,	//u_torso
		0.5,	0.5,	0.5,	//l_torso
		0,		0,		0,		//r_shoulder
		0,		0,		0,		//l_shoulder
		0,		0,		0,		//r_hip
		0,		0,		0,		//l_hip
		0,		0,		0,		//r_arm
		0,		0,		0,		//r_forearm
		0.8,	0.8,	0.8,	//r_hand
		0,		0,		0,		//l_arm
		0,		0,		0,		//l_forearm
		0.8,	0.8,	0.8,	//l_hand
		0,		0,		0,		//r_thigh
		0,		0,		0,		//r_calf
		0.8,	0.8,	0.8,	//r_foot
		0,		0,		0,		//l_thigh
		0,		0,		0,		//l_calf
		0.8,	0.8,	0.8		//l_foot
};

int call_num = 0;
pose_estimator::pose_estimator(int frame_width, int frame_height, int noDisparities)
:rotation_max(90,45,45),
 rotation_min(-90,-45,-45),
 enable_bones(false),
 frame_width(frame_width),
 frame_height(frame_height),
 numberOfDisparities(noDisparities)
{
	srand(time(0));
	cv::theRNG().state = time(0);
	std::cout << "[Pose Estimator] New Pose estimator: " << frame_width << "x" << frame_height << std::endl;
	load_param();

	input_depth = cv::Mat::zeros(frame_height, frame_width, CV_8UC1);
	input_frame = cv::Mat::zeros(frame_height, frame_width, CV_8UC1);
	window_boundaries = cv::Mat::zeros(frame_height, frame_width, CV_8UC1);
	cv::rectangle(window_boundaries,cv::Point(1,1),cv::Point(frame_width-2,frame_height-2),cv::Scalar(255),2);

	model = new ogre_model(frame_width,frame_height);
	pose_predictor = new pose_prediction(1);
	set_modeler_depth();
	camera_viewspace = model->get_camera_viewspace();

	//init_particles(true);
#if DEBUG_WIND_POSE
	cv::namedWindow("input frame");
	cv::namedWindow("best global silhouette");
	cv::namedWindow("best global depth");
	cv::namedWindow("best global diff");
	cv::namedWindow("test");
#endif

#if USE_GPU
	std::cout << "Begin creating ocl context..." << std::endl;
	std::vector<cv::ocl::Info> ocl_info;
	int devnums=getDevice(ocl_info);
	std::cout << "End creating ocl context...\n" << std::endl;
	if(devnums<1){
		std::cout << "no OPENCL device found\n";
	}
#endif
}

pose_estimator::~pose_estimator()
{
	cv::destroyAllWindows();
	delete(model);
	delete(pose_predictor);
}

/**
 * Load camera parameters
 */
inline void pose_estimator::load_param()
{
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
		//bone_min = cv::Mat::zeros(19,3,CV_32FC1);
		//bone_max = cv::Mat::ones(19,3,CV_32FC1)*180;
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
inline void pose_estimator::set_modeler_depth()
{
	std::cout << "[Pose Estimator] Setting model depth"<< std::endl;
	cv::Mat limits = calculate_depth_limit();
	model->set_depth_limits(limits.at<double>(2),limits.at<double>(1),limits.at<double>(0));
	model->set_camera_clip(limits.at<double>(2)/2,2*limits.at<double>(0));
	std::cout<<"Depth set to "<<limits.at<double>(2)<<","<<limits.at<double>(1)<<","<<limits.at<double>(0)<<std::endl;
}

/**
 * Calculates and returns the min, max and center depth value calculated by the stereo rig
 * on the center of the left camera. This is done with respect to the real camera's world
 * coordinate system, so its in 'mm'.
 */
inline cv::Mat pose_estimator::calculate_depth_limit()
{
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
	disparity_limit.at<uchar>(2,2) = numberOfDisparities;
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
 * Initializes swarm particles and the best position. If start_over is true, then everything resets to random.
 * If start_over is false, then only velocities are initiated
 */
inline void pose_estimator::init_particles()
{
	std::cout << "[Pose Estimator] Initializing/Reseting particles"<< std::endl;
	evolution_num = 0;
	score_change_count = 0;

	switch(mode){
	case FIRST_RUN :
		if(human_position==cv::Point3f())human_position = estimate_starting_position();
		//Since we are in detection mode, search for user facing camera
		rotation_max = cv::Point3f(15,15,15);
		rotation_min = cv::Point3f(-15,-15,-15);
		std::cout<< "[Pose Estimator] estimated starting position" << human_position <<std::endl;

		//Initialize best particle position
		best_global_position.bones_rotation = cv::Mat(19,3,CV_32FC1,H_pose);
		//best_global_position.bones_rotation = cv::Mat::zeros(19,3,CV_32FC1);
		best_global_position.scale = 800;
		best_global_position.model_position = cv::Point3f(human_position.x,human_position.y,human_position.z);
		best_global_position.model_rotation = cv::Point3f(0,0,0);

	case SEARCH :

		//reset particle position predictor
		pose_predictor->reset_predictor();
		enable_bones = false;

		break;
	case TRACK :
		enable_bones = true;
		//		rotation_max = cv::Point3f(90,45,45);
		//		rotation_min = cv::Point3f(-90,-45,-45);
		rotation_max = cv::Point3f(180,45,45);
		rotation_min = cv::Point3f(-180,-45,-45);
		best_global_position_predicted = pose_predictor->predict(best_global_position);		//Predict where the best particle position should be
		std::cout<<"predicted rotation"<<best_global_position_predicted.model_rotation<<std::endl;
		break;
	}

	std::vector<ogre_model::particle_position> list(1);
	list[0] = best_global_position;
	best_global_depth = model->get_depth(list)[0].clone();	//Refresh depth and extremas best solution
	best_global_depth = 255-best_global_depth;
	best_global_extremas = model->get_extremas()[0].clone();

	particle previous_best_particle;
	previous_best_particle.particle_depth = best_global_depth.clone();			//Last best disparity
	previous_best_particle.extremas = best_global_extremas.clone();				//Last best extremas
	previous_best_particle.current_position = best_global_position;
	std::cout<<previous_best_particle.current_position.bones_rotation.size()<<std::endl;
	best_global_score = calc_score(previous_best_particle);						//Calculate score of last best position on new given frame

	for(int i = 0; i<swarm_size; i++){
		if(mode==FIRST_RUN)swarm[i].id = i;
		swarm[i].reset_array[0]=false;
		swarm[i].reset_array[1]=false;
		swarm[i].reset_array[2]=false;
		swarm[i].reset_array[3]=false;
		swarm[i].obsolete = false;
		init_single_particle(swarm[i], ALL, false  );
	}

}

/**
 * Initializes a single particle. Usefull when initializing for the first time
 * or reseting individual particles.
 * @param	type	POSITION,ROTATION,BONES,SCALE,ALL marks the type of initialization
 * @param	fine	Boolean flag. If true, only the next position gets initialized.
 */
inline void pose_estimator::init_single_particle(particle& singleParticle, int type, bool fine)
{
	//singleParticle.x=0;					//'x' variable is responsible for forcing the particle to converge to a local solution
	if( type==POSITION || type==ALL ){
		if(!fine){
			if(mode==TRACK)singleParticle.current_position.model_position = best_global_position_predicted.model_position + cv::Point3f(rand_gen.gaussian(0.3)*300,rand_gen.gaussian(0.3)*300,rand_gen.gaussian(0.3)*300) ;
			else singleParticle.current_position.model_position = get_random_model_position(false);
		}
		singleParticle.next_position.model_position = get_random_model_position(true);
		if(!fine)singleParticle.best_position.model_position = singleParticle.current_position.model_position;
		//singleParticle.position_violation = cv::Mat::ones(3,1,CV_32FC1);
	}

	if( type==ROTATION || type==ALL ){
		if(!fine){
			if(mode==TRACK)singleParticle.current_position.model_rotation = best_global_position_predicted.model_rotation + cv::Point3f(rand_gen.gaussian(0.3)*5,rand_gen.gaussian(0.3)*5,rand_gen.gaussian(0.3)*5);
			else singleParticle.current_position.model_rotation = get_random_model_rot(false);
		}
		singleParticle.next_position.model_rotation = get_random_model_rot(true);
		if(!fine)singleParticle.best_position.model_rotation =  singleParticle.current_position.model_rotation;
		//singleParticle.rotation_violation = cv::Mat::ones(3,1,CV_32FC1);
	}

	if( type==BONES || type==ALL ){
		if(!fine){
			if(mode==TRACK)singleParticle.current_position.bones_rotation = best_global_position_predicted.bones_rotation.clone() + get_random_bones_rot(true);
			else singleParticle.current_position.bones_rotation = get_random_bones_rot(false).clone();
		}
		singleParticle.next_position.bones_rotation = get_random_bones_rot(true).clone();
		if(!fine)singleParticle.best_position.bones_rotation = singleParticle.current_position.bones_rotation.clone();
		//singleParticle.bones_violation = cv::Mat::ones(19,3,CV_32FC1);
	}

	if( type==SCALE || type==ALL ){
		if(!fine){
			if(mode==FIRST_RUN) singleParticle.current_position.scale = round(600 + rand_gen.uniform(0.f,1.0f)*(600));
			else singleParticle.current_position.scale = best_global_position.scale;
		}
		singleParticle.next_position.scale = round(rand_gen.uniform(-0.5f,0.5f)*400);
		if(!fine)singleParticle.best_position.scale = singleParticle.current_position.scale;
	}
	if( type==ALL ){
		//TODO maybe I should remove this or initialize it differently
		singleParticle.best_score = 0;
		singleParticle.x=0;
	}
}

/**
 * Estimate where the human is relative to the camera
 */
inline cv::Point3f pose_estimator::estimate_starting_position()
{
	cv::Mat result, disparity, input_silhouette;
	input_depth.convertTo(disparity, CV_32FC1,numberOfDisparities/255.);
	cv::reprojectImageTo3D(disparity, result, Q);
	threshold(input_depth,input_silhouette,1,255,cv::THRESH_BINARY);
	cv::Scalar mean_depth = mean(result, input_silhouette);
	return cv::Point3f(-mean_depth.val[0], -mean_depth.val[1], mean_depth.val[2]);
}

/**
 * Calculates random values for the bones based on loaded movement
 * restrictions.
 */
inline cv::Mat pose_estimator::get_random_bones_rot(bool velocity)
{
	float weights[]={
			2,	2,	2,		//head
			1,	1,	1,		//u_torso
			1,	1,	1,		//l_torso
			0,	1,	1,		//r_shoulder
			0,	1,	1,		//l_shoulder
			0,	1,	1,		//r_hip
			0,	1,	1,		//l_hip
			4,	3,	3,		//r_arm
			3,	0,	4,		//r_forearm
			2,	2,	2,		//r_hand
			4,	3,	3,		//l_arm
			3,	0,	4,		//l_forearm
			2,	2,	2,		//l_hand
			2,	3,	3,		//r_thigh
			0,	3,	0,		//r_calf
			0,	2,	0,		//r_foot
			2,	3,	3,		//l_thigh
			0,	3,	0,		//l_calf
			0,	2,	0		//l_foot
	};


	cv::Mat mat_weights(19,3,CV_32FC1,weights);

	if(enable_bones){
		cv::Mat result;
		result = get_random_19x3_mat(velocity).clone();
		if(velocity){
			result = 0.3*(bone_max-bone_min).mul(result.mul(mat_weights/4));
			//std::cout<<result<<std::endl;
			//result = 60*result.mul(mat_weights/4);
			//result = 40*result;
		}else{
			result = bone_max.mul(result) + bone_min.mul(1-result);
			//result = 360*result;
		}
		return result;
	}else{
		return cv::Mat(19,3,CV_32FC1,H_pose);
		return cv::Mat::zeros(19,3,CV_32FC1);
	}
}

/**
 * Returns a random 19x3 mat, using gaussian or uniform distribution
 */
inline cv::Mat pose_estimator::get_random_19x3_mat(bool velocity){
	cv::Mat result = cv::Mat::zeros(19,3,CV_32FC1);
	for(int i=0 ; i<19 ; i++){
		for(int j=0 ; j<3 ; j++){
			if(velocity){
				result.at<float>(i,j) = (float)rand_gen.gaussian(0.3);
				//result.at<float>(i,j) = (float)(rand_gen.uniform(0.f,2.1f)-1);
			}else{
				result.at<float>(i,j) = (float)rand_gen.uniform(0.f,1.0f);		//uniform distribution
			}
		}
	}
	return result;
}

/**
 * Calculates random values for model position.
 * @param	velocity	Boolean flag. If true, then a gaussian distribution is used instead of a uniform one.
 * @param	min_z		Minimum z distance that the model may be found.
 * @param 	max_z		Maximum z distance that the model may be found.
 */
inline cv::Point3f pose_estimator::get_random_model_position(bool velocity){
	cv::Point3f position;

	if(velocity){
		position.x = rand_gen.gaussian(0.3) * 300 ;	//gaussian distribution
		position.y = rand_gen.gaussian(0.3) * 300 ;
		position.z = rand_gen.gaussian(0.3) * 300 ;
	}else{
		position.x = human_position.x + rand_gen.uniform(-1.0f,1.0f) * 300 ;	//uniform distribution
		position.y = human_position.y + rand_gen.uniform(-1.0f,1.0f) * 300 ;
		position.z = human_position.z + rand_gen.uniform(-1.0f,1.0f) * 300 ;
	}
	return position;
}

/**
 * Calculates min and max valuse of x and y for given value of z
 */
inline cv::Mat pose_estimator::get_MinMax_XY(float z){
	cv::Mat result = cv::Mat(4,1,CV_32FC1);

	result.at<float>(0,0) = (camera_viewspace(4,0).x - camera_viewspace(0,0).x)/(camera_viewspace(4,0).z-camera_viewspace(0,0).z)*z + (camera_viewspace(0,0).x*camera_viewspace(4,0).z - camera_viewspace(0,0).z*camera_viewspace(4,0).x)/(camera_viewspace(4,0).z-camera_viewspace(0,0).z);	//min_x
	result.at<float>(1,0) = (camera_viewspace(5,0).x - camera_viewspace(1,0).x)/(camera_viewspace(4,0).z-camera_viewspace(0,0).z)*z + (camera_viewspace(1,0).x*camera_viewspace(4,0).z - camera_viewspace(0,0).z*camera_viewspace(5,0).x)/(camera_viewspace(4,0).z-camera_viewspace(0,0).z);	//max_x
	result.at<float>(2,0) = (camera_viewspace(6,0).y - camera_viewspace(2,0).y)/(camera_viewspace(4,0).z-camera_viewspace(0,0).z)*z + (camera_viewspace(2,0).y*camera_viewspace(4,0).z - camera_viewspace(0,0).z*camera_viewspace(6,0).y)/(camera_viewspace(4,0).z-camera_viewspace(0,0).z);	//min_y
	result.at<float>(3,0) = (camera_viewspace(5,0).y - camera_viewspace(1,0).y)/(camera_viewspace(4,0).z-camera_viewspace(0,0).z)*z + (camera_viewspace(1,0).y*camera_viewspace(4,0).z - camera_viewspace(0,0).z*camera_viewspace(5,0).y)/(camera_viewspace(4,0).z-camera_viewspace(0,0).z);	//max_y

	return result;
}

/**
 * Calculates random values for model global rotation
 */
inline cv::Point3f pose_estimator::get_random_model_rot(bool velocity){
	float yaw, pitch, roll;
	if(velocity){
		yaw = rand_gen.gaussian(0.3)*0.3*(rotation_max.x - rotation_min.x);
		pitch = rand_gen.gaussian(0.3)*0.3*(rotation_max.y - rotation_min.y);
		roll = rand_gen.gaussian(0.3)*0.3*(rotation_max.z - rotation_min.z);
	}else{
		yaw = rotation_min.x + rand_gen.uniform(0.f,1.1f)*(rotation_max.x - rotation_min.x);
		pitch =  rotation_min.y + rand_gen.uniform(0.f,1.1f)*(rotation_max.y - rotation_min.y);
		roll =  rotation_min.z + rand_gen.uniform(0.f,1.1f)*(rotation_max.z - rotation_min.z);
	}
	return cv::Point3f(yaw, pitch, roll);
}

/**
 *Refreshes the scores and thus evolving the swarm
 */
inline void pose_estimator::calc_evolution(){
	bool best_solution_updated;
	score_change_count = 0;
	evolution_num = 0;

	std::deque<double> time_deque;

	while(evolution_num<MAX_EVOLS && score_change_count<MAX_SCORE_CHANGE_COUNT){
		cv::theRNG().state = time(0);
		evolution_num++;
		best_solution_updated = false;
		//double t = (double)cv::getTickCount();

		//Call modeler to generate poses
		std::vector<ogre_model::particle_position> list(swarm_size);
		for(int i=0;i<swarm_size;i++){
			list[i]=swarm[i].current_position;
		}
		cv::Mat* depth_list = model->get_depth(list);
		cv::Mat_<cv::Point>* extremas_list =  model->get_extremas();
		for(int i=0;i<swarm_size;i++){
			swarm[i].particle_depth = depth_list[i].clone();
			swarm[i].extremas = extremas_list[i];
		}


#if NUM_THREADS>1
		//parallel evaluation of particles
		for(int i=0;i<NUM_THREADS-1;i++){
			thread_group.create_thread(boost::bind(&pose_estimator::evaluate_particles, this, i*swarm_size/NUM_THREADS, (i+1)*swarm_size/NUM_THREADS));
		}
		thread_group.create_thread(boost::bind(&pose_estimator::evaluate_particles, this, (NUM_THREADS-1)*swarm_size/NUM_THREADS, swarm_size));
		thread_group.join_all();
#else
		//serial evaluation of particles
		evaluate_particles(0,swarm_size);
#endif


		//update the best solution
		int best_pos = 0;
		for(int i=0;i<swarm_size;i++){
			if(swarm[i].best_score > best_global_score){
				best_solution_updated = true;
				score_change_count = 0;
				best_global_score = swarm[i].best_score;
				best_global_position = swarm[i].best_position;
				best_pos = i;
			}
		}
		//update global best depth and extremas
		if(best_solution_updated){
			best_global_extremas = swarm[best_pos].extremas.clone();
			best_global_depth = swarm[best_pos].particle_depth.clone();
			human_position = best_global_position.model_position;
		}


		if(!best_solution_updated)score_change_count++;	//count how many time the score hasn't change
		obsolete_counter = 0;							//initialize obsolete counter before counting

		//evolution of particles
		for(int i=0;i<swarm_size;i++){
			calc_next_position(swarm[i]);
			if(swarm[i].obsolete){
				obsolete_counter++;
			}
		}

		//Bones will be enabled after all particles are rendered obsolete at least once
		if(obsolete_counter==swarm_size && !enable_bones && mode!=FIRST_RUN){
			enable_bones = true;
			for(int i = 0; i<swarm_size; i++){
				init_single_particle(swarm[i], BONES, false);
			}
		}

		std::cout<< "[Pose Estimator]evol:"<<evolution_num<<" best global score: " <<best_global_score << " best global position: " << best_global_position.model_position <<" best global scale: "<<best_global_position.scale <<std::endl;

		//		t = ((double)cv::getTickCount() - t)*1000./cv::getTickFrequency();
		//		time_deque.push_back(t);
		//		//if((int)time_deque.size()>40)time_deque.pop_front();
		//		double mean = 0;
		//		for(int i=0; i<(int)time_deque.size(); i++)mean+=time_deque[i];
		//		std::cout<<"[Pose Estimator]Mean execution time: "<<mean/(time_deque.size()?time_deque.size():1)<<"ms"<<std::endl;

#if DEBUG_WIND_POSE
		//Remove this if not needed. It wastes 18ms!
		get_instructions();
		show_best_solution();
#endif
	}

}

/**
 * Evaluates the given range of particles, updating their score and best position accordingly.
 */
inline void pose_estimator::evaluate_particles(int start, int end){
	for(int i=start;i<end;i++){
		swarm[i].particle_depth = 255 - swarm[i].particle_depth;	//Inverting disparity map;

		//double t = (double)cv::getTickCount();
		double score = calc_score(swarm[i]);
		//std::cout<<"[Pose Estimator]Mean execution time: "<<((double)cv::getTickCount() - t)*1000./cv::getTickFrequency()<<"ms"<<std::endl;

		if(score > swarm[i].best_score){
			swarm[i].best_score = score;
			if(enable_bones)swarm[i].best_position.bones_rotation = swarm[i].current_position.bones_rotation.clone();
			swarm[i].best_position.model_position = swarm[i].current_position.model_position;
			swarm[i].best_position.model_rotation = swarm[i].current_position.model_rotation;
			swarm[i].best_position.scale = swarm[i].current_position.scale;
		}
	}
}

/**
 *Calculates the best next position for the given particle, based on
 *random movement, its best position and the global best one
 */
inline void pose_estimator::calc_next_position(particle& particle_inst){
	round_position(particle_inst.current_position);
	cv::Mat particle_inst_old_bones = particle_inst.current_position.bones_rotation.clone();
	cv::Point3f particle_inst_old_position = particle_inst.current_position.model_position;
	cv::Point3f particle_inst_old_rotation = particle_inst.current_position.model_rotation;
	float particle_inst_old_scale = particle_inst.current_position.scale;
	float w1 = A/exp(particle_inst.x);

	particle_inst.next_position.model_position.x = w1*particle_inst.next_position.model_position.x + c1*rand_gen.uniform(0.f,1.0f)*(particle_inst.best_position.model_position.x - particle_inst.current_position.model_position.x) + c2*rand_gen.uniform(0.f,1.0f)*(best_global_position.model_position.x - particle_inst.current_position.model_position.x);
	particle_inst.next_position.model_position.y = w1*particle_inst.next_position.model_position.y + c1*rand_gen.uniform(0.f,1.0f)*(particle_inst.best_position.model_position.y - particle_inst.current_position.model_position.y) + c2*rand_gen.uniform(0.f,1.0f)*(best_global_position.model_position.y - particle_inst.current_position.model_position.y);
	particle_inst.next_position.model_position.z = w1*particle_inst.next_position.model_position.z + c1*rand_gen.uniform(0.f,1.0f)*(particle_inst.best_position.model_position.z - particle_inst.current_position.model_position.z) + c2*rand_gen.uniform(0.f,1.0f)*(best_global_position.model_position.z - particle_inst.current_position.model_position.z);

	particle_inst.next_position.model_rotation.x = w1*particle_inst.next_position.model_rotation.x + c1*rand_gen.uniform(0.f,1.0f)*(particle_inst.best_position.model_rotation.x - particle_inst.current_position.model_rotation.x) + c2*rand_gen.uniform(0.f,1.0f)*(best_global_position.model_rotation.x - particle_inst.current_position.model_rotation.z);
	particle_inst.next_position.model_rotation.y = w1*particle_inst.next_position.model_rotation.y + c1*rand_gen.uniform(0.f,1.0f)*(particle_inst.best_position.model_rotation.y - particle_inst.current_position.model_rotation.y) + c2*rand_gen.uniform(0.f,1.0f)*(best_global_position.model_rotation.y - particle_inst.current_position.model_rotation.y);
	particle_inst.next_position.model_rotation.z = w1*particle_inst.next_position.model_rotation.z + c1*rand_gen.uniform(0.f,1.0f)*(particle_inst.best_position.model_rotation.z - particle_inst.current_position.model_rotation.z) + c2*rand_gen.uniform(0.f,1.0f)*(best_global_position.model_rotation.z - particle_inst.current_position.model_rotation.z);

	if(enable_bones){
		particle_inst.next_position.bones_rotation = w1*particle_inst.next_position.bones_rotation + c1*get_random_19x3_mat(false).mul(particle_inst.best_position.bones_rotation - particle_inst.current_position.bones_rotation) + c2*get_random_19x3_mat(false).mul(best_global_position.bones_rotation-particle_inst.current_position.bones_rotation);
	}
	particle_inst.next_position.scale = w1*particle_inst.next_position.scale + c1*rand_gen.uniform(0.f,1.1f)*(particle_inst.best_position.scale - particle_inst.current_position.scale) + c2*rand_gen.uniform(0.f,1.1f)*(best_global_position.scale - particle_inst.current_position.scale);

	particle_inst.current_position.model_position += particle_inst.next_position.model_position;
	particle_inst.current_position.model_rotation += particle_inst.next_position.model_rotation;
	if(enable_bones)add(particle_inst.current_position.bones_rotation,particle_inst.next_position.bones_rotation,particle_inst.current_position.bones_rotation);
	particle_inst.current_position.scale += particle_inst.next_position.scale;

	//Place the new position within limits
#if NUM_THREADS > 1
	human_position_mutex.lock();
	cv::Point3f local_human_position = human_position;
	human_position_mutex.unlock();
#else
	cv::Point3f local_human_position = human_position;
#endif
	if(particle_inst.current_position.model_position.x < local_human_position.x - 1000){
		particle_inst.current_position.model_position.x = local_human_position.x - 1000;
		//init_single_particle(particle_inst,POSITION,true);
		//particle_inst.next_position.model_position.x = -0.1*particle_inst.next_position.model_position.x;
		//particle_inst.next_position.model_position.x =0;
	}else if(particle_inst.current_position.model_position.x > local_human_position.x + 1000){
		particle_inst.current_position.model_position.x = local_human_position.x + 1000;
		//init_single_particle(particle_inst,POSITION,true);
		//particle_inst.next_position.model_position.x = -0.1*particle_inst.next_position.model_position.x;
		//particle_inst.next_position.model_position.x =0;
	}

	if(particle_inst.current_position.model_position.y < local_human_position.y - 1000){
		particle_inst.current_position.model_position.y = local_human_position.y - 1000;
		//init_single_particle(particle_inst,POSITION,true);
		//particle_inst.next_position.model_position.y = -0.1*particle_inst.next_position.model_position.y;
		//particle_inst.next_position.model_position.y =0;
	}else if(particle_inst.current_position.model_position.y > local_human_position.y + 1000){
		particle_inst.current_position.model_position.y = local_human_position.y + 1000;
		//init_single_particle(particle_inst,POSITION,true);
		//particle_inst.next_position.model_position.y = -0.1*particle_inst.next_position.model_position.y;
		//particle_inst.next_position.model_position.y =0;
	}

	if(particle_inst.current_position.model_position.z < local_human_position.z - 1000){
		particle_inst.current_position.model_position.z = local_human_position.z - 1000;
		//init_single_particle(particle_inst,POSITION,true);
		//particle_inst.next_position.model_position.z = -0.1*particle_inst.next_position.model_position.z;
		//particle_inst.next_position.model_position.z =0;
	}else if(particle_inst.current_position.model_position.z > local_human_position.z + 1000){
		particle_inst.current_position.model_position.z = local_human_position.z + 1000;
		//init_single_particle(particle_inst,POSITION,true);
		//particle_inst.next_position.model_position.z = -0.1*particle_inst.next_position.model_position.z;
		//particle_inst.next_position.model_position.z =0;
	}

	//Place the new rotation within limits
	if( particle_inst.current_position.model_rotation.x > rotation_max.x){	//yaw
		particle_inst.current_position.model_rotation.x = rotation_max.x;
		//init_single_particle(particle_inst,ROTATION,true);
		//particle_inst.next_position.model_rotation.x = -0.3*particle_inst.next_position.model_rotation.x;
	}else if(particle_inst.current_position.model_rotation.x < rotation_min.x){
		particle_inst.current_position.model_rotation.x = rotation_min.x;
		//init_single_particle(particle_inst,ROTATION,true);
		//particle_inst.next_position.model_rotation.x = -0.3*particle_inst.next_position.model_rotation.x;
	}

	if(particle_inst.current_position.model_rotation.y > rotation_max.y){	//pitch
		particle_inst.current_position.model_rotation.y = rotation_max.y;
		//init_single_particle(particle_inst,ROTATION,true);
		//particle_inst.next_position.model_rotation.y = -0.3*particle_inst.next_position.model_rotation.y;
	}else if(particle_inst.current_position.model_rotation.y < rotation_min.y){
		particle_inst.current_position.model_rotation.y = rotation_min.y;
		//init_single_particle(particle_inst,ROTATION,true);
		//particle_inst.next_position.model_rotation.y = -0.3*particle_inst.next_position.model_rotation.y;
	}

	if(particle_inst.current_position.model_rotation.z > rotation_max.z){	//roll
		particle_inst.current_position.model_rotation.z = rotation_max.z;
		//init_single_particle(particle_inst,ROTATION,true);
		//particle_inst.next_position.model_rotation.z = -0.3*particle_inst.next_position.model_rotation.z;
	}else if(particle_inst.current_position.model_rotation.z < rotation_min.z){
		particle_inst.current_position.model_rotation.z = rotation_min.z;
		//init_single_particle(particle_inst,ROTATION,true);
		//particle_inst.next_position.model_rotation.z = -0.3*particle_inst.next_position.model_rotation.z;
	}

	//Place the new scale within limits
	if(particle_inst.current_position.scale < 600){
		particle_inst.current_position.scale = 600;
		//init_single_particle(particle_inst,SCALE,true);
		//particle_inst.next_position.scale = -0.3*particle_inst.next_position.scale;
	}else if (particle_inst.current_position.scale >1200){
		particle_inst.current_position.scale = 1200;
		//init_single_particle(particle_inst,SCALE,true);
		//particle_inst.next_position.scale = -0.3*particle_inst.next_position.scale;
	}

	//Place the new bones rotation within limits
	//cv::Mat result = particle_inst.current_position.bones_rotation.clone();
	cv::min(particle_inst.current_position.bones_rotation,bone_max,particle_inst.current_position.bones_rotation);
	cv::max(particle_inst.current_position.bones_rotation,bone_min,particle_inst.current_position.bones_rotation);
	//particle_inst.bones_violation = (result == particle_inst.current_position.bones_rotation);	//Check if before and after limiting
	//particle_inst.bones_violation.convertTo(particle_inst.bones_violation, CV_32FC1,(float)1.0/255);

	//particle_inst.bones_violation -= 0.0;
	//particle_inst.next_position.bones_rotation = particle_inst.next_position.bones_rotation.mul(particle_inst.bones_violation);
	//particle_inst.next_position.bones_rotation.copyTo(particle_inst.next_position.bones_rotation);

	//Round current_position, since its float
	round_position(particle_inst.current_position);

	//Reset particles, if they have settled on some dimensions and on the best_global_position
	if( particle_inst.current_position.model_position == particle_inst_old_position){
#if DEBUG_COUT_POSE
		std::cout<< "Particle " << particle_inst.id <<" resets: POSITION"<< std::endl;
#endif
		init_single_particle(particle_inst,POSITION,true);
		particle_inst.reset_array[0] = true;
	}
	if(particle_inst.current_position.model_rotation == particle_inst_old_rotation){
#if DEBUG_COUT_POSE
		std::cout<< "Particle " << particle_inst.id <<" resets: ROTATION"<< std::endl;
#endif
		init_single_particle(particle_inst,ROTATION,true);
		particle_inst.reset_array[1] = true;
	}
	if(enable_bones && (cv::countNonZero(particle_inst.current_position.bones_rotation == particle_inst_old_bones)>30)){
#if DEBUG_COUT_POSE
		std::cout<< "Particle " << particle_inst.id <<" resets: BONES"<< std::endl;
#endif
		init_single_particle(particle_inst,BONES,false);
		particle_inst.reset_array[2] = true;
	}
	if( particle_inst.current_position.scale == particle_inst_old_scale){
#if DEBUG_COUT_POSE
		std::cout<< "Particle " << particle_inst.id <<" resets: SCALE"<< std::endl;
#endif
		init_single_particle(particle_inst,SCALE,true);
		//particle_inst.best_score = 0;
		particle_inst.reset_array[3] = true;
	}

	if(particle_inst.reset_array[0] && particle_inst.reset_array[1] && particle_inst.reset_array[2] && particle_inst.reset_array[3])particle_inst.obsolete = true;

	if(particle_inst.x<log(10*A)){
		particle_inst.x+=log(10*A)/N;
	}else{
#if DEBUG_COUT_POSE
		//std::cout<< "Particle " << particle_inst.id <<" resets: ALL "<< std::endl;
#endif
		particle_inst.x = 0;
		//init_single_particle(particle_inst,ALL,true);
		particle_inst.obsolete = true;
	}
}

/**
 * Rounds particle position
 */
inline void pose_estimator::round_position(ogre_model::particle_position& position){
	position.model_position.x = floor(position.model_position.x + 0.5);
	position.model_position.y = floor(position.model_position.y + 0.5);
	position.model_position.z = floor(position.model_position.z + 0.5);
	position.model_rotation.x = floor(position.model_rotation.x + 0.5);
	position.model_rotation.y = floor(position.model_rotation.y + 0.5);
	position.model_rotation.z = floor(position.model_rotation.z + 0.5);
	position.bones_rotation.convertTo(position.bones_rotation, CV_32SC1);
	position.bones_rotation.convertTo(position.bones_rotation, CV_32FC1);
	position.scale = floor(position.scale + 0.5);
}

/**
 *Calculates fitness score
 */
inline double pose_estimator::calc_score(const particle& particle_inst){
	//float dist1 = 1./(1 + sqrt(pow((particle_inst.extremas.at<ushort>(0,0)-input_head.x),2) + pow((particle_inst.extremas.at<ushort>(0,1)-input_head.y),2)));
	double right_dist = 0, left_dist = 0;
	bool found_rhand=false, found_lhand=false;
	if(right_hand!=cv::Point(-1,-1)) {
		right_dist = (1 - (pow((particle_inst.extremas.at<cv::Point>(9).x-right_hand.x),2) + pow((particle_inst.extremas.at<cv::Point>(9).y-right_hand.y),2))/90000);
		if(right_dist>1)right_dist=1;
		if(right_dist<0)right_dist=0;
		found_rhand = true;
	}
	if(left_hand!=cv::Point(-1,-1)) {
		left_dist = (1 - (pow((particle_inst.extremas.at<cv::Point>(12).x-left_hand.x),2) + pow((particle_inst.extremas.at<cv::Point>(12).y-left_hand.y),2))/90000);
		if(left_dist>1)left_dist=1;
		if(left_dist<0)left_dist=0;
		found_lhand = true;
	}

#if !USE_GPU
	cv::Mat fuzzy_mat, diff_mat;

	cv::absdiff(particle_inst.particle_depth, input_depth,diff_mat);
	//cv::min(diff_mat,200,diff_mat);
	cv::bitwise_xor(particle_inst.particle_depth, input_depth,fuzzy_mat);

	//	cv::Mat temp, temp2;
	//	cv::max(particle_inst.particle_depth,input_depth,temp);	//or
	//	cv::min(particle_inst.particle_depth,input_depth,temp2); //and

	cv::Rect box = cv::boundingRect(particle_inst.extremas);
//	box.x -= box.width/6;
//	box.width += box.width/3;
//	box.y -= box.height/6;
//	box.height += box.height/3;
//	box &= cv::Rect(0,0,frame_width, frame_height);

	cv::Mat bones_pen= cv::Mat(19,3,CV_32FC1,bones_penalty);

	cv::Mat bones_score = particle_inst.current_position.bones_rotation.mul(bones_pen)/360;

	//cv::Mat silhouette;
	//cv::threshold(particle_inst.particle_depth, silhouette, 1, 255, CV_THRESH_BINARY);
	//cv::bitwise_and(temp,silhouette,temp);
	//cv::bitwise_and(temp2,silhouette,temp2);

	double size_score = (double)(box & user_bounds).area()/(box | user_bounds).area();
	size_score = (1-pow(1-size_score,2));

	//cv::Mat penalty_mask = particle_inst.particle_depth.clone();
	//cv::rectangle(penalty_mask,user_bounds,cv::Scalar(0),CV_FILLED);

	double final_score = size_score  *(1 - mean(diff_mat, particle_inst.particle_depth).val[0]/255.)* (1 - mean(fuzzy_mat).val[0]/255.);//*(1-cv::mean(bones_score).val[0]);
	if(found_rhand&&found_lhand&&enable_bones){
		return final_score * right_dist*left_dist;
	}else if(found_rhand&&enable_bones){
		return  final_score * right_dist;
	}else if(found_lhand&&enable_bones){
		return  final_score * left_dist;
	}else{
		//std::cout<<(1 - mean(fuzzy_mat).val[0]/fuzzy_max)<<" " <<1-cv::mean(distance_penalty,penalty_mask).val[0]/100<<" "<<(1 - mean(diff_mat).val[0]/100)<<std::endl;
		//		cv::Mat edges_particle;
		//		Canny( particle_inst.particle_depth, edges_particle, 50, 150, 3 );
		//		double edge_dist = edge_estimator.calculate_edge_distance(input_frame, edges_particle,2);
		//		return edge_dist; (1 - mean(fuzzy_mat).val[0]/fuzzy_max)
		//*(1 - mean(diff_mat).val[0]/diff_max)
		//* (1-cv::mean(distance_penalty,penalty_mask).val[0]/100) ;
		return  final_score ;
		//return (1 - mean(fuzzy_mat).val[0]/fuzzy_max)  * (1-cv::mean(distance_penalty,penalty_mask).val[0]/100) ;

	}
#else
	cv::ocl::oclMat ocl_input_depth;
	cv::ocl::oclMat ocl_particle_depth;
	cv::ocl::oclMat result;
	cv::Scalar ocl_mean, ocl_stddev;
	ocl_input_depth.upload(input_depth);
	ocl_particle_depth.upload(particle_inst.particle_depth);
	cv::ocl::bitwise_xor(ocl_input_depth,ocl_particle_depth,result);
	cv::ocl::meanStdDev(result,ocl_mean, ocl_stddev);
	return 1./(1+ocl_mean.val[0]) * 1./(1+cv::mean(distance_penalty,particle_inst.particle_depth).val[0]);
#endif

}

/**
 * Image viewer of best solution, for debugging purposes
 */
void pose_estimator::show_best_solution(){
	cv::Mat local_input = input_depth.clone();
	cv::rectangle(local_input,user_bounds,cv::Scalar(255));
	imshow("input frame", local_input);
	cv::Mat result;

	bitwise_xor(input_depth, best_global_depth, result);
	cv::Mat jet_depth_map2;

	cv::applyColorMap(best_global_depth, jet_depth_map2, cv::COLORMAP_JET );

	cv::rectangle(jet_depth_map2,get_model_bound_rect(),cv::Scalar(255,255,255));

	imshow("best global depth", jet_depth_map2);

	std::stringstream ss;
	ss << "Frame No: " <<call_num << " evolution No: "<<evolution_num;
	cv::putText(result,ss.str(),cv::Point(5,20),0,0.5,cv::Scalar(127),2);
	imshow("best global diff", result);

	imshow("particle 0",swarm[0].particle_depth);
	imshow("particle 1",swarm[1].particle_depth);
}

/**
 * Keystroke handler, for debugging purposes
 */
void pose_estimator::get_instructions(){
	int key = cv::waitKey(1) & 255;
	if ( key == 27 ) exit(1);	//Esc
	if ( key == 114 ) { //R
		human_position=cv::Point3f();
		mode = FIRST_RUN;
		init_particles();

	}
}

/*
 * Finds and returns best global solution of pose estimator.
 */
particle pose_estimator::find_pose(cv::Mat disparity_frame, pose_estimator::operation_mode mode, cv::Rect local_user_bounds,cv::Point left_marker, cv::Point right_marker)
{
	double t = (double)cv::getTickCount();
	this->mode = mode;
	user_bounds = local_user_bounds;
	left_hand = left_marker;
	right_hand =right_marker;

	distance_penalty = cv::Mat::ones(disparity_frame.size(), CV_8UC1)*255;
	cv::rectangle(distance_penalty,user_bounds,cv::Scalar(0),CV_FILLED);
	distanceTransform(distance_penalty, distance_penalty, CV_DIST_L2, 3);
	//cv::threshold(distance_penalty,distance_penalty,50,255,CV_THRESH_TRUNC);
	cv::normalize(distance_penalty, distance_penalty, 0, 100, cv::NORM_MINMAX, CV_32FC1);
	//imshow("distance_penalty",distance_penalty);

	//Canny( color_frame, input_frame, 50, 150, 3 );
	disparity_frame.copyTo(input_depth);
	switch(mode){
	case FIRST_RUN:
		MAX_SCORE_CHANGE_COUNT=10;
		break;
	case SEARCH:
		MAX_SCORE_CHANGE_COUNT=30;
		break;
	case TRACK:
		MAX_SCORE_CHANGE_COUNT=20;
		break;
	}
	init_particles();
	calc_evolution();

	call_num++;

	t = ((double)cv::getTickCount() - t)*1000./cv::getTickFrequency();
	std::cout<< "[Pose Estimator]Swarm searching finished. Time:"<< t << "s Best global score: " <<best_global_score << " best global position: " << best_global_position.model_position <<" best global scale: "<<best_global_position.scale <<std::endl;

	particle final_solution;
	final_solution.particle_depth = best_global_depth.clone();
	final_solution.extremas = best_global_extremas.clone();
	final_solution.best_score = best_global_score;
	final_solution.best_position = best_global_position;
	return final_solution;
}

#undef DEBUG_WIND_POSE
#undef DEBUG_COUT_POSE
#undef NUM_THREADS
#undef USE_GPU
