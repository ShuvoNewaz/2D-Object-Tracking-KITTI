import torch
from src.models.retinanet.outputs import retinanet_outputs


def predict(test_image, model):
    device = test_image.device
    test_image_tensor = torch.unsqueeze(test_image, 0)

    with torch.no_grad():
        test_image_tensor = test_image_tensor.float()
        features, classification, regression, anchors = model(test_image_tensor)

        results = retinanet_outputs(model, test_image_tensor,
                                    classification,
                                    regression, anchors)[0]
        for key in results:
            results[key] = results[key].cpu()
        scores, predictedLabels, predictedBoxes = \
                    results["scores"], results["labels"], results["boxes"]
        
        # Plot the bounding boxes over the image
        keep_boxes = scores > 0.35
    
    return predictedBoxes[keep_boxes].to(device), predictedLabels[keep_boxes].to(device)