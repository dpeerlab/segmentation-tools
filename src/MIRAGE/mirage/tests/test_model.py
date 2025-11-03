import pytest
import cv2
import mirage

def test_model():
    import numpy as np
    import cv2
    import skimage

    # assert data loading errors
    pytest.raises(FileNotFoundError, mirage.tl.get_data, "sample0.tiff")  # file not found
    pytest.raises(ValueError, mirage.tl.get_data, "sample0.png")  # png not supported

    # read image
    print("Reading image...")
    img_grey = mirage.tl.get_data("sample1.tiff")

    # distort image
    print("Distorting image...")
    mat_dist = np.array([[0.98, 0.02, 5], [-0.01, 1, 5], [0, 0, 1]])
    img_dist = skimage.transform.warp(img_grey, mat_dist)

    # setup mask
    bin_mask = np.ones(img_grey.shape)

    # create model
    print("Creating model...")
    #tf.random.set_seed(123) --> still getting different results
    mirage_model = mirage.MIRAGE(
        images=img_grey,
        references=img_dist,
        bin_mask=bin_mask,
        pad=12,
        offset=12,
        num_neurons=196,
        num_layers=2,
        pool=1,
        loss="SSIM"
    )

    # train model
    print("Training model...")
    mirage_model.train(batch_size=256, num_steps=256, lr__sched=True, LR=0.005)

    # calculate transformation
    print("Calculating transformation...")
    mirage_model.compute_transform()
    img_tran = mirage_model.apply_transform(img_dist)

    # evaluate metrics
    print("Evaluating metrics...")
    ssim_dist, simg_dist = skimage.metrics.structural_similarity(img_grey, img_dist, data_range=1, full=True)
    ssim_tran, simg_tran = skimage.metrics.structural_similarity(img_grey, img_tran, data_range=1, full=True)

    # assert result
    print("Asserting results...")
    mirage.exceptions.assert_metric(0.6516, ssim_dist, "Input SSIM", abs=0.01)
    mirage.exceptions.assert_metric(0.9642, ssim_tran, "Transformed SSIM", abs=0.01)
    
    # visually inspect
    mesh = mirage.tl.Mesh(80, 160, pad=35)
    mirage.pl.plot_before_after(img_grey, img_dist, img_tran, mesh=mesh)