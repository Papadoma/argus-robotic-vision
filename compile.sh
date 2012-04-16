#!/bin/bash

g++ stereo_vision.cpp `pkg-config opencv --cflags` -o stereo_vision.out `pkg-config opencv --libs`
