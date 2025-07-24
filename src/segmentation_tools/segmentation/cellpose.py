# from cellpose import models


def run_cellpose(image):
    model = models.Cellpose(gpu=True, model_type="nuclei")
    masks, *_ = model.eval(image, diameter=None, channels=[0, 0])
    return masks
