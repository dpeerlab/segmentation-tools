# MIRAGE

Efficient and scalable image registration (alignment) for multiplexed imaging data. 

⚠️ Currently only available for GPU.

## Installation

```
pip install cellmirage
```

## Example

Also, check out the example in `tests/quickstart.ipynb`:

Before alignment:

![before](media/vis/before_registration.png)

After alignment:

![after](media/vis/after_registration.png)


```
# load package
import mirage

# Load images
image = mirage.tl.get_data("sample2_image.tiff")
reference = mirage.tl.get_data("sample2_reference.tiff")
# Images must be 2D numpy arrays (grayscale only) and scaled to 0-1.

# Initialise model
mirage_model = mirage.MIRAGE(
    images=image,
    references=reference,
    bin_mask=bin_mask,
    pad=12,
    offset=12,
    num_neurons=196,  # more for larger images
    num_layers=2,  # more for larger images
    pool=1, 
    loss="SSIM"
)

# Train model
mirage_model.train(batch_size=256, num_steps=256, lr__sched=True, LR=0.005)

# Apply transformation
mirage_model.compute_transform()
image_aligned = mirage_model.apply_transform(image)

# Inspect results pre/post alignment of a smale mesh
mesh = mirage.tl.Mesh(x=80, y=160, pad=35)
mirage.pl.plot_before_after(reference, image, image_aligned, mesh=mesh)
```
