#!/bin/bash
datadir=dataset
kitti_home_page=https://s3.eu-central-1.amazonaws.com/avg-kitti
training_labels=data_tracking_label_2.zip
images=data_tracking_image_2.zip
mkdir -p "${datadir}"

declare -a items_to_download=("${training_labels}"
                                "${images}"
                            )
for item in "${items_to_download[@]}";
do
    wget "${kitti_home_page}/${item}"
    unzip "${item}" -d "${datadir}"
    rm "${item}"
done