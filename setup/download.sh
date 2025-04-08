#!/bin/bash
datadir=dataset
pretrained_dir=pretrained
saved_resnet_dir="${pretrained_dir}/resnet"
saved_obj_detector_dir="${pretrained_dir}/object_detector"
kitti_home_page=https://s3.eu-central-1.amazonaws.com/avg-kitti

training_labels=data_tracking_label_2.zip
images=data_tracking_image_2.zip

for dir in "${datadir}" "${pretrained_dir}" \
            "${saved_resnet_dir}" "${saved_obj_detector_dir}";
do
    mkdir -p "${dir}"
done

# Download and organize the dataset
declare -a items_to_download=("${training_labels}"
                                "${images}"
                            )
for item in "${items_to_download[@]}";
do
    wget "${kitti_home_page}/${item}"
    unzip "${item}" -d "${datadir}"
    rm "${item}"
done

# Download and organize the pretrained weights

## ResNet
declare -A resnet_links
resnet_links["resnet18"]="https://download.pytorch.org/models/resnet18-5c106cde.pth"
resnet_links["resnet50"]="https://download.pytorch.org/models/resnet50-19c8e357.pth"
resnet_links["resnet101"]="https://download.pytorch.org/models/resnet101-5d3b4d8f.pth"
resnet_links["resnet152"]="https://download.pytorch.org/models/resnet152-b121ed2d.pth"

declare -A resnet_local_names
resnet_local_names["resnet18"]="resnet18.pth"
resnet_local_names["resnet50"]="resnet50_v2.pth"
resnet_local_names["resnet101"]="resnet101_v2.pth"
resnet_local_names["resnet152"]="resnet152_v2.pth"

for resnet in "${!resnet_links[@]}";
do
    wget -O "${saved_resnet_dir}/${resnet_local_names[$resnet]}"\
    "${resnet_links[${resnet}]}"
done

## Object Detector
gdown --fuzzy https://drive.google.com/file/d/1wGPepHVjsJY17BFRvqpQCpFSbQL0tMVm/view?usp=sharing
mv object_detector_2d.pt "${saved_obj_detector_dir}/checkpoint.pt"