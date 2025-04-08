import torch
from PIL import Image
from src.bounding_box.utils import get_centers, match_boxes_iou
from src.bounding_box.predict import predict as detect_objects
from src.filters.kalman import *
from src.filters.particle import ParticleFilter
from src.plots.gif import *
import os
from torchvision import transforms as T
from torchvision.transforms.functional import to_pil_image

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
transform = T.Compose([T.ToTensor(),
                       T.Resize((384, 1248))]) # To avoid error in the network

def index_flat_to_2d(numCol, ind):

    return ind // numCol, ind % numCol


def nearest_center(old_centers, new_centers, threshold):
    """
    Calculate the distance of the object in question to the newly detected centers.
    Pick the new center that is closest to the old center.
    """
    # print("old", old_centers)
    # print("new", new_centers)
    distances = torch.cdist(old_centers, new_centers, p=2)

    # Multiple new boxes may be closest to a single old box, but only one can
    # belong to the old box. The distances will have to be compared globally instead
    # of box-wise.

    sorted_indices = distances.flatten().argsort()
    assigned_old = []
    assigned_new = []
    for ind in sorted_indices:
        r, c = index_flat_to_2d(len(new_centers), ind)
        if (r not in assigned_old and
            c not in assigned_new and
            distances[r, c] < threshold):
            assigned_old.append(r)
            assigned_new.append(c)
    index_map = []
    if len(assigned_old) > 0:
        index_map = torch.stack([torch.stack(assigned_old),
                                torch.stack(assigned_new)], dim=1)

    return index_map


def add_filter_dict(filterList, filterID,
                    filter_type, center, **filter_params):
    """
    Add new filter for newly detected objects.
    args:
        filterList:     Dictionary for filter parameters.
        filterID:       Unique identifier for the filter.
        filter_type:    The type of filter used (Kalman/EKF/Particle, etc.).
        center:         The state at which the filter is initialized.
    """
    filterList[filterID] = {}
    if filter_type == "kalman":
        filterList[filterID]["filter"] = KalmanFilter(**filter_params)
    elif filter_type == "particle":
        filterList[filterID]["filter"] = ParticleFilter(**filter_params)
    filterList[filterID]["filter"].initialize(center)


def no_new_measurement_update(filterList, filterIDs,
                         allowed_misses, gif_images,
                         image, ind):
    """
    Adds missed detection count to existing filters if no new bounding
    box is detected. If number of misses exceed the allowed number,
    the filter is removed.
    """
    if len(filterList) == 0:
        gif_images.append(to_pil_image(image))
    for filterID in filterIDs:
        filterList[filterID]["missed"] = \
            filterList[filterID].get("missed", 0) + 1
        # Update GIF
        update_gif_with_center(gif_images, image, ind, filterList, filterID)
        if filterList[filterID]["missed"] > allowed_misses:
            # Remove ID if missed for a few frames
            filterList.pop(filterID)
        

def run_filter_loop(images_dir, image_name_list, model,
                    filterList, allowed_misses,
                    threshold, first_image, filter_type,
                    **filter_params):
    """
    Runs objects detector and tracker on every frame. Updates the objects coordinates
    as necessary, and appends images to a GIF for visualization.
    args:
        images_dir:         The directory of the image frames.
        image_name_list:    The list of image names inside the image directory.
        model:              The object detector.
        filterList:         Dictionary for filter parameters.
        allowed_misses:     The number of consecutive measurement misses allowed
                            for each filter.
        threshold:          The upper bound of the distance between the old center
                            and new center for them to be a match.
        first_image:        The image to which all subsequent images are added
                            for the GIF.
        filter_type:        The type of filter used (Kalman/EKF/Particle, etc.)
    returns:
        Nothing. Creates a GIF in ./results
    """
    # Create results directory
    os.makedirs("results", exist_ok=True)
    gif_images = []

    # Run prediction step on existing filters
    for i, image_name in enumerate(image_name_list):
        image_dir = os.path.join(images_dir, image_name)

        # Store old centers for later comparison
        old_centers = torch.zeros((len(filterList), 2),
                                  dtype=torch.float32).to(device)
        filterIDs = list(filterList.keys())

        # Run prediction on existing filters
        for k, filterID in enumerate(filterIDs):
            filterList[filterID]["filter"].predict()
            old_centers[k] = filterList[filterID]["filter"].estimate().flatten()[:2]

        # Load image
        image = Image.open(image_dir)
        image = transform(image)
        image = image.to(device)

        # Run object detection
        boxes, labels = detect_objects(image, model)

        # Add missed count if no new measurements detected
        if len(labels) == 0:
            no_new_measurement_update(filterList, filterIDs,
                                      allowed_misses, gif_images,
                                      image, i)
            continue

        new_centers = get_centers(boxes)
        
        index_map = nearest_center(old_centers, new_centers, threshold)
        if len(index_map) > 0:
            # At least one new measurement corresponds to an old object
            old_object_indices_matched = index_map[:, 0]
            new_object_indices_matched = index_map[:, 1]
        else:
            # No new measurement corresponds to an old object
            old_object_indices_matched = []
            new_object_indices_matched = []

        # Check if any of the newly detected boxes belong to old objects
        # Run update step if new box matches with old object
        iterator = 0
        for k, filterID in enumerate(filterIDs):
            # If previous center k is not matched with new measurement
            if k not in old_object_indices_matched:
                filterList[filterID]["missed"] = \
                    filterList[filterID].get("missed", 0) + 1
            # Update step if matched
            else:
                filterList[filterID]["missed"] = 0 # Reset misses
                filterList[filterID]["filter"].update(new_centers[new_object_indices_matched[iterator]])
                if filter_type == "particle":
                    filterList[filterID]["filter"].resample()
                iterator += 1

            ## Update GIF using centers from prediction/update step
            update_gif_with_center(gif_images, image, i, filterList, filterID)

            ## Remove filter if missed for a few frames
            if filterList[filterID]["missed"] > allowed_misses:
                filterList.pop(filterID)
            
        for j in range(len(labels)):
            # Initialize new detections if new box didn't match with old objects
            if j not in new_object_indices_matched:
                ## Assign a new ID to new filter
                existingIDs = list(filterList.keys())
                try:
                    filterID = max(existingIDs) + 1 
                except:
                    filterID = 0

                ## Initialize Filter
                add_filter_dict(filterList,
                                filterID,
                                filter_type,
                                new_centers[j],
                                **filter_params)

                ## Update GIF
                update_gif_with_center(gif_images, image, i, filterList, filterID)

    # Save GIF
    first_image.save(f"results/{filter_type}.gif",
                    save_all=True,
                    append_images=gif_images,
                    durations=100,
                    loop=0)
    
    # Resize GIF
    resize_gif(f"results/{filter_type}.gif")
    
    # Open GIF
    os.system(f"open results/{filter_type}.gif")


def get_filter_params(filter_type, dt=0.1, sigma_ax=20, sigma_ay=5,
                      num_particles=1000, pos_std=20, process_std_x=50,
                      process_std_y=20, obs_std=50):
    """
    args:
        filter_type:    kalman or particle.
        dt:             Time between frames (Kalman)
        sigma_ax:       Process noise standard deviation in x-direction (Kalman).
        sigma_ay:       Process noise standard deviation in y-direction (Kalman).
        num_particles:  Number of particles (Particle).
        pos_std:        Postion standard deviation for initial particles (Particle).
        process_std_x:  Process noise standard deviation in x-direction (Particle).
        process_std_y:  Process noise standard deviation in y-direction (Particle).
        obs_std:        Observation noise standard deviation (Particle).
    """
    # Filter parameters
    if filter_type == "kalman":
        # Kalman Filter. Use larger standard deviations for faster moving objects.
        filter_params = {"dt": dt, "sigma_ax": sigma_ax, "sigma_ay":sigma_ay}
    elif filter_type == "particle":
        # Particle Filter. Use larger standard deviations for faster moving objects.
        filter_params = {"num_particles": num_particles, "pos_std": pos_std,
                         "process_std_x": process_std_x, "process_std_y": process_std_y,
                         "obs_std": obs_std}
    
    return filter_params