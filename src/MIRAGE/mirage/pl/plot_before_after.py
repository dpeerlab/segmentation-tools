from matplotlib import pyplot as plt
from mirage.pl.utils import get_overlay

def plot_before_after(ref, dist, tran, figsize=(12, 24), mesh=None):
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
        tran = mesh.get_mesh(tran)
    
    # get overlays
    dist_overlay = get_overlay(ref, dist)
    tran_overlay = get_overlay(ref, tran)
    
    # plot 
    plt.figure(figsize=figsize)
    
    plt.subplot(1, 3, 1)
    plt.imshow(ref, cmap='gray')
    plt.title('Reference')

    plt.subplot(1, 3, 2)
    plt.imshow(dist_overlay)
    plt.title('Distorted')

    plt.subplot(1, 3, 3)
    plt.imshow(tran_overlay)
    plt.title('Transformed')