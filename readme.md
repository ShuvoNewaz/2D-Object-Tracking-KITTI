# 2D Object Detection

## Project Environment

This repository is tested on Ubuntu 22 with an Nvidia RTX 4090 GPU. To use this repository, please follow these steps:

- Open terminal in your preferred work directory and enter the following commands:

  `git clone https://https://github.com/ShuvoNewaz/2D-Object-Tracking-KITTI`

  `cd 2D-Object-Tracking-KITTI`

- Make sure [Anaconda](https://www.anaconda.com/docs/getting-started/anaconda/install) is installed. Enter the following command:

  `conda env create -f setup/environment.yml`

This will create a conda environment with the required libraries. Activate the environment by typing

  `conda activate object_tracking_2d`

- Download and organize the dataset by entering

  `bash setup/download.sh`

  The above command will also download pretrained weights required for a subproblem (object detection). Refer to this [repository](https://github.com/ShuvoNewaz/2D-Object-Detection-KITTI) to get your own trained weight.

## Dataset

The dataset used for this project is the [KITTI 2D Tracking Evaluation](https://www.cvlibs.net/datasets/kitti/eval_object.php?obj_benchmark=2d). The parts of this dataset used are:
1. RGB Images (15 GB)
2. Training Labels (9 MB) (optional)

If the above steps were successfully followed, the dataset will already be downloaded to your required directory. The labels are optional, since we are using pretrained weights from a different model.

### Details of the Dataset

#### Images

The RGB images are continuous frames from different videos.

#### Labels

The labels exist only for the training set. These `.txt` files contain the following:

1. The object class.
2. Truncation - A measure of how much the object out of image bounds. 0 $\rightarrow$ non-truncated. 1 $\rightarrow$ truncated.
3. Occlusion. 0 $\rightarrow$ not occluded. 1 $\rightarrow$ partly occluded. 2 $\rightarrow$ mostly occluded. 3 $\rightarrow$ unknown.
4. Observation angle $[-\pi,\pi]$.
5. 2D bounding box of objects in the image. Contains coordinates of 4 corners.
6. 3D object dimensions in meters.

    i. Height
    
    ii. Width
    
    iii. length
7. 3D center location in camera coordinates.
8. Rotation angle around the y-axis $[-\pi,\pi]$.

For the 2D object tracking task, we only need the object class and the bounding box coordinates.

## Model and Data Processing

### Object Detection

The details on the object detector can be found [here]((https://github.com/ShuvoNewaz/2D-Object-Detection-KITTI)).

### Filters

#### Kalman Filter

The state to be determined is,

$$ \textbf{x} = 
\begin{bmatrix}

x \\
y \\
\dot{x} \\
\dot{y}

\end{bmatrix}
$$

where $(x,y)$ are the center coordinates of a bounding box and $(\dot{x},\dot{y})$ are the velocities of the box center in 2 dimensions. The state transition matrix models how the state evolves from one frame to the next. Assuming a constant velocity model,

$$ \textbf{A} = 
\begin{bmatrix}

1 & 0 & \Delta t & 0 \\
0 & 1 & 0 & \Delta t \\
0 & 0 & 1 & 0 \\
0 & 0 & 0 & 1

\end{bmatrix}
$$

where $\Delta t$ is the time step. Consider the following equations of motion:

$$\textbf{x}_{t+1} = \dot{\textbf{x}}_t \Delta t + \frac{1}{2} \textbf{a}_t \Delta t^2$$

$$\dot{\textbf{x}}_{t+1} = \dot{\textbf{x}}_t + \textbf{a}_t \Delta t$$

The acceleration is modeled as a zero-mean Gaussian process noise with variance $\Sigma_a$,

$$ a_t \sim \mathcal{N} \left(
  
  \begin{bmatrix}
  0 \\ 0
  \end{bmatrix},\Sigma_a
  \right)
$$

where

$$
\Sigma_a =
\begin{bmatrix}
  \sigma_{a_x}^2 & 0 \\
  0 & \sigma_{a_y}^2
\end{bmatrix}
$$

The process noise vector arises purely from acceleration,

$$
\textbf{w} =
\begin{bmatrix}

\frac{1}{2} \Delta t^2 a_x \\
\frac{1}{2} \Delta t^2 a_y \\
\Delta t \cdot a_x \\
\Delta t \cdot a_y

\end{bmatrix}
$$

Now, the full 2D process noise covariance matrix models the uncertainty in the process,

$$\textbf{Q} = \mathbb{E}[\textbf{w} \textbf{w} ^\mathrm{T}]$$

$$\textbf{Q} =
\begin{bmatrix}
\frac{1}{4} \Delta t^4 \sigma_{a_x}^2 & 0 & \frac{1}{2} \Delta t^3 \sigma_{a_x}^2 & 0 \\
0 & \frac{1}{4} \Delta t^4 \sigma_{a_y}^2 & 0 & \frac{1}{2} \Delta t^3 \sigma_{a_y}^2 \\
\frac{1}{2} \Delta t^3 \sigma_{a_x}^2 & 0 & \Delta t^2 \sigma_{a_x}^2 & 0 \\
0 & \frac{1}{2} \Delta t^3 \sigma_{a_y}^2 & 0 & \Delta t^2 \sigma_{a_y}^2
\end{bmatrix}
$$

##### Initialization

Initialize a random error covariance matrix $\textbf{P}[n|n-1] \in \mathbb{R}^{4 \times 4}$

##### Prediction Step

The new estimated state is,

$$\hat{\textbf{x}}[n|n-1] = \textbf{A} \cdot \hat{\textbf{x}}[n-1|n-1]$$

The new error covariance matrix is,

$$\textbf{P}[n|n-1] = \textbf{A}[n|n-1] \cdot \textbf{P}[n-1|n-1] \cdot \textbf{A}^{\textrm{T}}[n|n-1] + \textbf{Q}[n]$$

##### Observation Step

The observation is the center of the bounding box,

$$ \textbf{z} = \begin{bmatrix} x_c \\ y_c \end{bmatrix} $$

which is obtained from the object detector. The observation matrix maps the predicted state to the observed measurement. Since the observation vector only contains position $(x_c, y_c)$, we want to extract those from the full state vector.

$$
\textbf{H} = 
\begin{bmatrix}
1 & 0 & 0 & 0 \\
0 & 1 & 0 & 0
\end{bmatrix}
$$

The measurement noise covariance matrix represents the uncertainty in the observation. It depends on the accuracy of the bounding box detection.

$$
\textbf{R} = 
\begin{bmatrix}
\sigma_{x_c}^2 & 0 \\
0 & \sigma_{y_c}^2
\end{bmatrix}
$$

The Kalman gain is,

$$\textbf{K} = \textbf{P[n|n-1]} \cdot \textbf{H}^\textrm{T} \cdot
(\textbf{H} \cdot \textbf{P}[n|n-1] \cdot \textbf{H}^{\textrm{T}} + \textbf{R}) ^ {-1}$$

The updated state is,

$$\hat{\textbf{x}}[n|n] = \hat{\textbf{x}}[n|n-1] + \textbf{K} \cdot (\textbf{z} - \textbf{H} \cdot \hat{\textbf{x}}[n|n-1])$$

The error covariance matrix is updated as,

$$\textbf{P}[n|n] = (\textbf{I} - \textbf{K} \cdot \textbf{H}) \cdot \textbf{P}[n|n-1]$$

## Results
