# 2D Object Tracking using Adaptive Filters

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

  The above command will also download pretrained weights required for a subproblem (object detection). Refer to this [repository](https://github.com/ShuvoNewaz/2D-Object-Detection-KITTI) to get your own trained weights.

- To view results, open the [notebook](notebook.ipynb), activate the installed environment, and run all.

## Dataset

The dataset used for this project is the [KITTI 2D Tracking Evaluation](https://www.cvlibs.net/datasets/kitti/eval_tracking.php). The parts of this dataset used are:
1. RGB Images (15 GB)
2. Training Labels (9 MB) (optional)

If the above steps were successfully followed, the dataset will already be downloaded to your required directory. The labels are optional for this project, since we are using pretrained weights from a different dataset.

### Details of the Dataset

#### Images

The RGB images are continuous frames from different videos. The time difference between consecutive frames is 0.1 seconds.

#### Labels

The labels exist only for the training set. For this work, we have not used the labels provided for the tracking dataset. Instead, we use the weights generated from this [repository](https://github.com/ShuvoNewaz/2D-Object-Detection-KITTI) to predict our own bounding boxes. 

## Object Detector

The details on the object detector can be found [here]((https://github.com/ShuvoNewaz/2D-Object-Detection-KITTI)).

## Filters

### Kalman Filter

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

#### Initialization

Initialize a random error covariance matrix $\textbf{P}[n|n-1] \in \mathbb{R}^{4 \times 4}$. This matrix indirectly models the randomness of the process.

#### Prediction Step

The new estimated state is,

$$\hat{\textbf{x}}[n|n-1] = \textbf{A} \cdot \hat{\textbf{x}}[n-1|n-1]$$

The new error covariance matrix is,

$$\textbf{P}[n|n-1] = \textbf{A} \cdot \textbf{P}[n-1|n-1] \cdot \textbf{A}^{\textrm{T}} + \textbf{Q}[n]$$

#### Update Step

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

### Particle Filter

Unlike the Kalman filter that estimates a single state, a particle filter maintains many guesses of the object's state. The Kalman filter tracks the mean and covariance of a state under the assumption of linear dynamics and Gaussian noise. The particle filter approximates the entire distribution using a set of samples called Particles. Each particle represents a hypothesis of where the tracked object might be. Each particle is a possible state, and each has a weight representing how likely it is. Over time, the filter 

- Predicts where each particle would move (based on a motion model + noise).
- Updates its weight based on how well it matches the current observation.
- Resamples - keeping the most likely particles and discarding unlikely ones.

#### Initialization

We start with a cloud of particles centered around the initial measurement (detection). If there are no initial detections, we create a cloud that is uniformly distributed.

#### Prediction

For each particle $i$, we apply a motion model. Assuming constant velocity added with random noise,

$$\textbf{x}_ {i}[n] = \textbf{x}_ {i}[n-1] + \dot{\textbf{x}} \cdot \Delta t + \mathcal{N}(0,\Sigma)$$

#### Update (Likelihood Estimation)

For each particle $i$, we compute how close it is to the detected measurement. The assigned likelihood is inversely proportional to this distance. The weights are all then normalized to get a sum of 1.

#### Resampling

We draw a new set of particles by sampling from the current set, with probability proportional to their weights. Particles with high likelihood are duplicated and with low likelihood are discarded.

#### State Estimation

The current estimated position is the weighted average of the particles.

## Results

The red dots represent the predicted position ($(x,y)$ coordinates) of the objects. When the object detector fails to detect the object (noisy measurement), the Kalman filter makes a guess of where the object might be using the predicted velocity. However, if the object detector fails to detect for a certain number of consecutive frames, we retire the filter to avoid spurious detections. The top GIF is for Kalman filters, and the bottom is for Particle filters.

<p align="center">
  <img src="results/kalman.gif"/>
</p>
<p align="center">
  <img src="results/particle.gif"/>
</p>