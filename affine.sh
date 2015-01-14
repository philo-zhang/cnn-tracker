#/use/bin/env sh

TOOLS=./build/tools

nohup $TOOLS/caffe train \
		--solver=cnn-tracker/solver_affine.prototxt > cnn-tracker/log &
