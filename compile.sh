#!/bin/bash

g++ stereo_vision.cpp `pkg-config opencv --cflags` -o stereo_vision.out `pkg-config opencv --libs`
g++ calculate_depth.cpp `pkg-config opencv --cflags` -o calculate_depth.out `pkg-config opencv --libs`
