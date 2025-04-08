import torch


def get_centers(boxes):
    """
    args:
        box: torch.tensor
    """
    x_c = (boxes[:, 0] + boxes[:, 2]) / 2
    y_c = (boxes[:, 1] + boxes[:, 3]) / 2
    numBoxes = len(boxes)

    return torch.cat([x_c.view(numBoxes, -1),
                      y_c.view(numBoxes, -1)], dim=1).to(boxes.device)


def get_corners(centers, width, height):
    x1 = centers[:, 0] - width / 2
    y1 = centers[:, 1] - height / 2
    x2 = centers[:, 0] + width / 2
    y2 = centers[:, 1] + height / 2
    numBoxes = len(x1)

    return torch.cat([x1.view(numBoxes, -1),
                      y1.view(numBoxes, -1),
                      x2.view(numBoxes, -1),
                      y2.view(numBoxes, -1)], dim=1).to(centers.device)


def match_boxes_iou(filter_boxes, detector_boxes):
    """
    args:
        filter_boxes:   (N, 4) array containing the coordinates
                        of the boxes predicted by the filters.
        detector_boxes: (M, 4) array containing the coordinates
                        of the boxes predicted by the object detector.
    returns:
        iou:            (N, M) array containing the intersection over
                        union of the filter boxes and the detector boxes.
    """
    N, M = len(filter_boxes), len(detector_boxes)
    x11, y11, x12, y12 = (filter_boxes[:, 0].unsqueeze(1),
                          filter_boxes[:, 1].unsqueeze(1),
                          filter_boxes[:, 2].unsqueeze(1),
                          filter_boxes[:, 3].unsqueeze(1))
    x21, y21, x22, y22 = (detector_boxes[:, 0].unsqueeze(0),
                          detector_boxes[:, 1].unsqueeze(0),
                          detector_boxes[:, 2].unsqueeze(0),
                          detector_boxes[:, 3].unsqueeze(0))

    intersection_x1 = torch.maximum(x11, x21)
    intersection_y1 = torch.maximum(y11, y21)
    intersection_x2 = torch.minimum(x12, x22)
    intersection_y2 = torch.minimum(y12, y22)

    intersection_area = (torch.clamp(intersection_x2 - intersection_x1, min=0) *
                         torch.clamp(intersection_y2 - intersection_y1, min=0))

    filter_boxes_area = (x12 - x11) * (y12 - y11)
    detector_boxes_area = (x22 - x21) * (y22 - y21)
    union_area = filter_boxes_area + detector_boxes_area - intersection_area

    iou = intersection_area / (union_area + 1e-6)  # Adding a small epsilon to avoid division by zero
    matched_boxes = iou.max(dim=1)[1]

    return matched_boxes