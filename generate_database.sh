#!/bin/bash

mkdir Database;
if [ $# -eq 0 ]; then
	echo "Expected argument: Path to the database."
else find $1/ -iname '*.avi' -exec ./bin/FaceDetector {} \;
fi
