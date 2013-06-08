#include "pose_predictor.hpp"

pose_prediction::pose_prediction(int choice){
	history_known = false;
	this->choice = choice;
}

inline void pose_prediction::naive(ogre_model::particle_position& new_pos){
	if(!history_known){
		position_old.model_position = new_pos.model_position;
		position_old.model_rotation = new_pos.model_rotation;
		position_old.bones_rotation = new_pos.bones_rotation.clone();
		position_old.scale = new_pos.scale;
		history_known = true;
	}else{
		ogre_model::particle_position pos_temp;
		pos_temp = new_pos;
		new_pos.model_position = (new_pos.model_position - position_old.model_position) + new_pos.model_position;
		new_pos.model_rotation = (new_pos.model_rotation - position_old.model_rotation) + new_pos.model_rotation;
		new_pos.bones_rotation = (new_pos.bones_rotation - position_old.bones_rotation) + new_pos.bones_rotation;
		new_pos.scale = (new_pos.scale - position_old.scale) + new_pos.scale;
		position_old = pos_temp;
	}
}

inline void pose_prediction::kalman(ogre_model::particle_position& new_pos){
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

void pose_prediction::reset_predictor(){
	history_known = false;
}

