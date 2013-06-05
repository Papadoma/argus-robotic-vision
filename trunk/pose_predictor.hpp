#ifndef POSE_PREDICTOR_HPP
#define POSE_PREDICTOR_HPP

#include <opencv2/opencv.hpp>

class pose_prediction{
private:
	int choice;
	ogre_model::particle_position position_old;
	bool history_known;

	void naive(ogre_model::particle_position&);
	void kalman(ogre_model::particle_position&);

public:
	pose_prediction(int);
	void predict(ogre_model::particle_position&);
};

pose_prediction::pose_prediction(int choice){
	history_known = false;
	this->choice = choice;
}

void pose_prediction::naive(ogre_model::particle_position& new_pos){
	if(!history_known){
		position_old.model_position = new_pos.model_position;
		position_old.model_rotation = new_pos.model_rotation;
		position_old.bones_rotation = new_pos.bones_rotation.clone();
		position_old.scale = new_pos.scale;
		history_known = true;
	}else{
		new_pos.model_position = (new_pos.model_position - position_old.model_position) + new_pos.model_position;
		new_pos.model_rotation = (new_pos.model_rotation - position_old.model_rotation) + new_pos.model_rotation;
		new_pos.bones_rotation = (new_pos.bones_rotation - position_old.bones_rotation) + new_pos.bones_rotation;
		new_pos.scale = (new_pos.scale - position_old.scale) + new_pos.scale;
	}
}

void pose_prediction::kalman(ogre_model::particle_position& new_pos){
	//TODO TO BE CONTINUED
}

void pose_prediction::predict(ogre_model::particle_position& cur_pos){
	switch(choice){
	case 0:
		naive(cur_pos);
		break;
	case 1:
		kalman(cur_pos);
		break;
	}
}

#endif
