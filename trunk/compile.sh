#!/bin/bash

#g++ stereo_vision.cpp `pkg-config opencv --cflags` -o stereo_vision.out `pkg-config opencv --libs`
#g++ calculate_depth.cpp `pkg-config opencv --cflags` -o calculate_depth.out `pkg-config opencv --libs`
#g++ stereo_match.cpp `pkg-config opencv --cflags` -o stereo_match.out `pkg-config opencv --libs`
g++ eye_stereo_match.cpp `pkg-config opencv --cflags` -o eye_stereo_match.out `pkg-config opencv --libs`
g++ -pthread parallel_eye_stereo_match.cpp `pkg-config opencv --cflags` -o parallel_eye_stereo_match.out `pkg-config opencv --libs`
#g++ stereo_calib3.cpp `pkg-config opencv --cflags` -o stereo_calib3.out `pkg-config opencv --libs`
#g++ capture_stereo_eye.cpp `pkg-config opencv --cflags` -o capture_stereo_eye.out `pkg-config opencv --libs`
g++ calibrate_stereo_eye.cpp `pkg-config opencv --cflags` -o calibrate_stereo_eye.out `pkg-config opencv --libs`
