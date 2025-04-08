from torchvision.transforms.functional import to_pil_image
from src.plots.coordinates import image_with_center_coordinates
from PIL import Image


def update_gif_with_center(image_list, image, ind,
                           filterList, filterID):
    coordinates = filterList[filterID]["filter"].estimate().\
                    flatten().cpu().numpy()[:2]
    if len(image_list) == ind:
        image_list.append(to_pil_image(image))
    image_list[ind] = image_with_center_coordinates(image_list[ind], coordinates)


def resize_gif(gif_path, new_size=(468, 144)):
    with Image.open(gif_path) as im:
        frames = []
        frame_counter = 0
        while True:
            try:
                im.seek(frame_counter)
                resized_frame = im.copy().resize(new_size, Image.LANCZOS)
                frames.append(resized_frame)
                frame_counter += 1
            except EOFError:
                break
        frames[0].save(gif_path,
                       save_all=True,
                       append_images=frames[1:],
                       durations=100,
                       loop=0)