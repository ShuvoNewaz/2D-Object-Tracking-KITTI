from PIL import ImageDraw
from torchvision.transforms.functional import pil_to_tensor, to_pil_image
import torch


def image_with_center_coordinates(image, centers):
    centers = centers.reshape(-1, 2)
    x, y = centers[:, 0], centers[:, 1]
    if torch.is_tensor(image):
        image = to_pil_image(image)
    draw = ImageDraw.Draw(image)
    for i in range(len(x)):
        draw.ellipse((x[i] - 10,
                      y[i] - 10,
                      x[i] + 10,
                      y[i] + 10), fill="red", outline="red")
    
    return image