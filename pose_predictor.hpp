#ifndef POSE_PREDICTOR_HPP
#define POSE_PREDICTOR_HPP

#include <opencv2/opencv.hpp>
#include "ogre_modeler.hpp"

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
	void reset_predictor();
};

#endif
