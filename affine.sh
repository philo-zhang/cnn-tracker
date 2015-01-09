#/use/bin/env sh

TOOLS=./build/tools

nohup $TOOLS/caffe train \
		--solver=cnn_tracker/solver_affine.prototxt > cnn_tracker/log &
