#/use/bin/env sh

TOOLS=./build/tools

$TOOLS/caffe train \
		--solver=cnn_tracker/solver.prototxt
