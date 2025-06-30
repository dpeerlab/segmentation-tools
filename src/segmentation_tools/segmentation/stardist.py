from stardist.models import StarDist2D


def run_stardist(image):
    model = StarDist2D.from_pretrained("2D_versatile_fluo")
    labels, _ = model.predict_instances(image)
    return labels
