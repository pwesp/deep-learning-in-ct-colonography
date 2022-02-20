#!/usr/bin/env bash

base="seg_cnn_"
ending=".index"
#ending=".data-00000-of-00001"

count=0

for i in ResNet18_3D_Dropout_SecondChannel_Segmentation_Ensemble_*.index; do
  echo "old filename: $i"
  
  newfilename="$base$count$ending"
  
  echo "new filename: $newfilename"
  
  mv $i $newfilename
  
  count=$((count+1))
done