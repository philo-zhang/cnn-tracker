#/use/bin/env sh

TOOLS=./build/tools

$TOOLS/caffe train \
		--solver=cnn-tracker/solver_affine_finetune.prototxt --weights=cnn-tracker/affine_iter_20000.caffemodel.backup --gpu=0
