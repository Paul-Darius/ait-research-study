#!/usr/bin/env sh

for name in Database/*/*.jpg; do
convert -resize 256x256\! $name $name
echo $name
done


caffe train --solver=src/siamese/siamese_mbk_solver.prototxt
