#!/bin/bash

find MBK_Videos/ -iname '*.avi' -exec ./bin/FaceDetector {} \;

