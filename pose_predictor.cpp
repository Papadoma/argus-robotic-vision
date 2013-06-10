#include "pose_predictor.hpp"

pose_prediction::pose_prediction(int choice){
	history_known = false;
	this->choice = choice;
}

inline ogre_model::particle_position pose_prediction::naive(const ogre_model::particle_position& cur_pos){
	if(!history_known){
		position_old.model_position = cur_pos.model_position;
		position_old.model_rotation = cur_pos.model_rotation;
		position_old.bones_rotation = cur_pos.bones_rotation.clone();
		position_old.scale = cur_pos.scale;
		history_known = true;
		return cur_pos;
	}else{
		ogre_model::particle_position pos_pred;
		pos_pred.model_position = (cur_pos.model_position - position_old.model_position) + cur_pos.model_position;
		pos_pred.model_rotation = (cur_pos.model_rotation - position_old.model_rotation) + cur_pos.model_rotation;
		pos_pred.bones_rotation = (cur_pos.bones_rotation - position_old.bones_rotation) + cur_pos.bones_rotation;
		pos_pred.scale = (cur_pos.scale - position_old.scale) + cur_pos.scale;
		position_old = cur_pos;
		return pos_pred;
	}
}

inline ogre_model::particle_position pose_prediction::kalman(const ogre_model::particle_position& new_pos){
	//TODO TO BE CONTINUED
}

ogre_model::particle_position pose_prediction::predict(const ogre_model::particle_position& cur_pos){
	switch(choice){
	case 0:
		return naive(cur_pos);
	case 1:
		return kalman(cur_pos);
	}
}

void pose_prediction::reset_predictor(){
	history_known = false;
}

