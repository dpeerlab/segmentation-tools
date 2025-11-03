import skimage.metrics as metrics
from matplotlib import pyplot as plt
from mirage.pl.utils import get_overlay

def plot_ssim(ref, dist, figsize=(12, 24), mesh=None):
    """
    Plot image: Reference, Distorted, and Transformed
    
    Parameters:
    * ref: Reference image
    * dist: Distorted image, or image to be aligned
    * tran: Transformed image, after applying MIRAGE transformations
    * mesh: Can be a Mesh instance to inspect specific part of image
    """
    
    if mesh is not None:
        ref = mesh.get_mesh(ref)
        dist = mesh.get_mesh(dist)
    
    ssim, ssim_image = metrics.structural_similarity(ref, dist, full=True, data_range=ref.max() - ref.min())
    
    # get overlays
    dist_overlay = get_overlay(ref, dist)
    
    # plot 
    plt.figure(figsize=figsize)
    
    plt.subplot(1, 3, 1)
    plt.imshow(ref, cmap='gray')
    plt.title('Reference')

    plt.subplot(1, 3, 2)
    plt.imshow(dist_overlay)
    plt.title('Distorted')

    plt.subplot(1, 3, 3)
    plt.imshow(ssim_image)
    plt.title(f"SSIM: {ssim:.4f}")