"""
Improvement ideas:
* Replace gaussian convolution with separable filters --> depthwise_conv2d
* Don't randomise coordinates, but rather use a grid
"""
from collections import defaultdict
import tensorflow as tf
import numpy as np
import skimage.transform
import warnings
from tqdm import trange


class MIRAGE_ORIGINAL(tf.keras.Model):
    def __init__(
        self,
        references,
        images,
        bin_mask=None,
        num_layers=3,
        num_neurons=1024,
        pad=24, 
        offset=24,
        pool=None,
        loss="SSIM",
        coeff=None,
    ):
        """
        Fit Mirage model.
        
        Parameters:
        * references: 2D array of images to be aligned. Can be a list of images.
        * images: 2D array of reference images. Can be a list of images.
        * bin_mask: 2D array of binary mask. Default is None which means all pixels are used.
        * num_layers: Number of layers in the neural network
        * num_neurons: Number of neurons in each layer
        * pad: Padding for each glimpse
        * offset: Maximum stepsize for each transformation
        * pool: Pooling factors reducing the image size
        * loss: Loss function, only SSIM is supported and tested for now
        * coeff: Coefficients for each image
        
        Methods:
        * train: Train the model
        * compute_transform: Compute the transformation for each pixel
        * apply_transform: Apply the transformation to an image
        
        Example:
        ```
        import mirage
        
        # Load image
        image = reference = mirage.tl.get_data("sample1.tiff")
        
        # Construct model
        mirage_model = mirage.MIRAGE(
            references=image,
            images=reference,
            pad=12,
            offset=12,
            num_neurons=196,
            num_layers=2,
            loss="SSIM"
        )

        # Train model
        mirage_model.train(batch_size=256, num_steps=256, lr__sched=True, LR=0.005)

        # Calculate transformation
        mirage_model.compute_transform()
        
        # Apply transformation
        img_tran = mirage_model.apply_transform(image)
        ```
        """
        
        super().__init__()

        # Set settings
        tf.config.run_functions_eagerly(True)

        # Assertions
        # * Images shape must align
        # * Images must be 2D or a list of 2D images
        # * Images must be between 0 and 1
        # * Mask must be same shape as images
        # * GPU must be available (for now)
        assert references.shape == images.shape, \
            f"Images must be same shape. Got {references.shape} (aligning) and " \
            f"{images.shape} (reference)."
        assert references.ndim in [2, 3], \
            f"Images must be 2D or a list of 2D images. Got dimension {references.ndim}."
        assert np.all(references >= 0) and np.all(references <= 1), \
            f"Image values must be between 0 and 1. Got min: {np.min(references)} " \
            f"and max: {np.max(references)}. Transform images to 0-1 using for example: `<img> / 255`"
        assert (bin_mask is None) or (bin_mask.shape == references.shape), \
            f"Mask must be same shape as images. Got {bin_mask.shape} (mask) and " \
            f"{references.shape} (images)."
        # TODO: CPU cannot support edges of meshes in function gather_nd() --> Only sample from "valid" region
        assert tf.config.list_physical_devices("GPU"), \
            f"GPU must be available. Got {tf.config.list_physical_devices('GPU')}." 
        if loss not in ["SSIM"]:
            warnings.warn(f"Loss function {loss} is not test. Recommended to use `SSIM` instead.")

        # Initialize
        self.num_layers = num_layers
        self.num_neurons = num_neurons
        self.references = references.astype("float32")
        self.images = images.astype("float32")
        self.bin_mask = bin_mask
        self.loss = loss            

        if self.references.ndim == 2:
            self.references = self.references[None, :, :]
            self.images = self.images[None, :, :]

        if self.bin_mask is None:
            self.bin_mask = np.ones(self.references.shape[1:3])

        self.num_images = self.references.shape[0]

        self.image_height = self.references.shape[1]
        self.image_width = self.references.shape[2]

        # Pooling to reduce image size
        if (pool is not None) and (pool > 1):
            self.pool = pool

            with tf.device("/CPU:0"):
                self.references = (
                    tf.squeeze(
                        (
                            tf.nn.avg_pool(
                                self.references.astype("float")[:, :, :, None],
                                ksize=[self.pool, self.pool],
                                strides=[self.pool, self.pool],
                                padding="SAME",
                            )
                        ),
                        axis=-1,
                    )
                    .numpy()
                    .astype("float32")
                )
                self.images = (
                    tf.squeeze(
                        (
                            tf.nn.avg_pool(
                                self.images.astype("float")[:, :, :, None],
                                ksize=[self.pool, self.pool],
                                strides=[self.pool, self.pool],
                                padding="SAME",
                            )
                        ),
                        axis=-1,
                    )
                    .numpy()
                    .astype("float32")
                )
                self.bin_mask = (
                    tf.squeeze(
                        (
                            tf.nn.avg_pool(
                                self.bin_mask.astype("float")[None, :, :, None],
                                ksize=[self.pool, self.pool],
                                strides=[self.pool, self.pool],
                                padding="SAME",
                            )
                        )
                    )
                    .numpy()
                    .astype("int32")
                )
        else:
            self.pool = 1

        self.offset = int(offset / self.pool)
        self.pad = int(pad / self.pool)

        if coeff is None:
            self.coeff = np.ones(self.num_images)
        else:
            self.coeff = np.asarray(coeff)

        self.layers_custom = []
        self.initializer = tf.keras.initializers.GlorotNormal()

        for i in range(self.num_layers):
            self.layers_custom.append(
                tf.keras.layers.Dense(
                    units=self.num_neurons,
                    kernel_initializer=self.initializer,
                    bias_initializer=self.initializer,
                )
            )
        self.vec_layer = tf.keras.layers.Dense(
            units=2,
            kernel_initializer=tf.keras.initializers.Zeros(),
            bias_initializer=tf.keras.initializers.Zeros(),
        )
        self.sig_layer = tf.keras.layers.Dense(
            units=1,
            kernel_initializer=tf.keras.initializers.Zeros(),
            bias_initializer=tf.keras.initializers.Zeros(),
        )

        self.y_mesh, self.x_mesh = tf.meshgrid(
            tf.range(-self.offset - self.pad, self.offset + self.pad + 1),
            tf.range(-self.offset - self.pad, self.offset + self.pad + 1),
        )

    @tf.function
    def image_loss(self, x, y):
        """
        Calculate the image loss
        
        x: Image 1
        y: Image 2
        """
        square_shape = tf.math.sqrt(tf.cast(x.shape[0], tf.float64))

        x = tf.reshape(x, [square_shape, square_shape])
        y = tf.reshape(y, [square_shape, square_shape])

        out = tf.reduce_mean(tf.image.ssim(x, y))
        out = tf.where(tf.math.is_nan(out), 0., out)
        return out

    @tf.function
    def multi_ssim(self, align_glimpses, ref_glimpses):
        """Deprecated"""
        warnings.warn(f"Deprecated. Use loss `SSIM` instead.", DeprecationWarning)
        align_glimpses = tf.reshape(
            tf.transpose(align_glimpses, [0, 3, 1, 2]),
            [-1, self.pad * 2 + 1, self.pad * 2 + 1],
        )
        ref_glimpses = tf.reshape(
            tf.transpose(ref_glimpses, [0, 3, 1, 2]),
            [-1, self.pad * 2 + 1, self.pad * 2 + 1],
        )

        if self.pad < 24:
            ssim_val = tf.reshape(
                tf.image.ssim_multiscale(
                    ref_glimpses[:, :, :, None],
                    align_glimpses[:, :, :, None],
                    max_val=1.0,
                    power_factors=[0.5, 0.5],
                ),
                [self.num_images, -1],
            )
        else:
            ssim_val = tf.reshape(
                tf.image.ssim_multiscale(
                    ref_glimpses[:, :, :, None],
                    align_glimpses[:, :, :, None],
                    max_val=1.0,
                    power_factors=[0.33, 0.33, 0.33],
                ),
                [self.num_images, -1],
            )

        return -tf.reduce_mean(ssim_val * self.coeff[:, None])

    @tf.function
    def SSIM(self, align_glimpses, ref_glimpses):
        align_glimpses = tf.reshape(
            tf.transpose(align_glimpses, [0, 3, 1, 2]),
            [-1, self.pad * 2 + 1, self.pad * 2 + 1],
        )
        ref_glimpses = tf.reshape(
            tf.transpose(ref_glimpses, [0, 3, 1, 2]),
            [-1, self.pad * 2 + 1, self.pad * 2 + 1],
        )

        ssim_val = tf.reshape(
            tf.image.ssim(
                ref_glimpses[:, :, :, None], align_glimpses[:, :, :, None], max_val=1.0
            ),
            [self.num_images, -1],
        )

        return -tf.reduce_mean(ssim_val * self.coeff[:, None])

    @tf.function
    def MSE(self, x, y):
        corr = tf.reduce_mean(tf.square(x - y))
        # return(tf.reduce_mean(tf.nn.softmax(y) * tf.math.log(tf.nn.softmax(x))))

        return corr

    @tf.function
    def norm_corr_loss(self, x, y):
        sum_mult = tf.reduce_sum(x * y, axis=-1)
        sum_square_x = tf.math.sqrt(tf.reduce_sum(tf.square(x), axis=-1))
        sum_square_y = tf.math.sqrt(tf.reduce_sum(tf.square(y), axis=-1))

        corr = tf.reduce_mean((sum_mult / (sum_square_x * sum_square_y)) * self.coeff)

        return -corr

    @tf.function
    def corr_loss(self, x, y):
        mean_x = tf.reduce_mean(x, axis=1)
        mean_y = tf.reduce_mean(y, axis=1)

        std_x = tf.math.reduce_std(x, axis=1)
        std_y = tf.math.reduce_std(y, axis=1)

        corr = tf.reduce_mean(
            (
                (x - mean_x[:, None])
                * (y - mean_y[:, None])
                / (std_x[:, None] * std_y[:, None])
            )
            * self.coeff[:, None]
        )
        # return(tf.reduce_mean(tf.nn.softmax(y) * tf.math.log(tf.nn.softmax(x))))

        return -corr

    @tf.function
    def softmax_loss(self, x, y):
        """
        Should be tensorflow native? For example: tf.keras.losses.CategoricalCrossentropy()
        """
        return tf.reduce_mean(tf.nn.softmax(y) * tf.math.log(tf.nn.softmax(x)))

    @tf.function
    def forward_pass(self, x_ind, y_ind):
        output = 2 * tf.stack([x_ind, y_ind], axis=1) - 1

        for i in range(self.num_layers):
            output = tf.nn.silu(self.layers_custom[i](output))

        vec_output = tf.nn.tanh(self.vec_layer(output))
        sig_output = tf.nn.softplus(self.sig_layer(output)) / 3 #"how sure"
        return (vec_output, sig_output)

    @tf.function
    def vec_to_field(self, vec, sig):
        x_dist = (
            tf.square(
                tf.cast(tf.linspace(-1, 1, 2 * self.offset + 1), dtype=tf.float32)[
                    None, :
                ]
                - vec[:, 0][:, None]
            )
            / sig
        )
        y_dist = (
            tf.square(
                tf.cast(tf.linspace(-1, 1, 2 * self.offset + 1), dtype=tf.float32)[
                    None, :
                ]
                - vec[:, 1][:, None]
            )
            / sig
        )

        field = x_dist[:, :, None] + y_dist[:, None, :]
        field = tf.reshape(
            tf.nn.softmax(-tf.reshape(field, [vec.shape[0], -1])),
            [vec.shape[0], 2 * self.offset + 1, 2 * self.offset + 1],
        )

        return field

    @tf.function
    def get_glimpses(self, x_ind, y_ind):
        X = (
            tf.tile(self.x_mesh[None, :, :], [x_ind.shape[0], 1, 1])
            + x_ind[:, None, None]
        )
        Y = (
            tf.tile(self.y_mesh[None, :, :], [y_ind.shape[0], 1, 1])
            + y_ind[:, None, None]
        )
        XY = tf.stack((X, Y), axis=-1)
        

        # for each image, get the XY pixels from references and concat them along axis 0
        pixel_glimpses_float = tf.concat(
            [
                tf.gather_nd(self.references[image_ind], XY)
                for image_ind in range(self.num_images)
            ],
            axis=0,
        )
        pixel_glimpses_float = tf.transpose(
            tf.stack(tf.split(pixel_glimpses_float, self.num_images, axis=0), axis=-1),
            [3, 1, 2, 0],
        )

        XY = XY[:, self.offset: -self.offset, self.offset: -self.offset, :]

        pixel_glimpses_ref = tf.concat(
            [
                tf.gather_nd(self.images[image_ind], XY)
                for image_ind in range(self.num_images)
            ],
            axis=0,
        )
        pixel_glimpses_ref = tf.transpose(
            tf.stack(tf.split(pixel_glimpses_ref, self.num_images, axis=0), axis=-1),
            [3, 1, 2, 0],
        )
        
        return pixel_glimpses_float, pixel_glimpses_ref

    @tf.function
    def compute_loss(self, output_offset, pixel_glimpses_float, pixel_glimpses_ref):
        pixel_glimpses_float = tf.nn.depthwise_conv2d(
            pixel_glimpses_float,
            tf.transpose(output_offset, [1, 2, 0])[:, :, :, None],
            strides=[1, 1, 1, 1],
            padding="VALID",
        )

        #

        if self.loss == "MultiSSIM":
            return self.multi_ssim(pixel_glimpses_float, pixel_glimpses_ref)

        if self.loss == "SSIM":
            return self.SSIM(pixel_glimpses_float, pixel_glimpses_ref)

        align_pixels = tf.reshape(pixel_glimpses_float, [self.num_images, -1])
        ref_pixels = tf.reshape(pixel_glimpses_ref, [self.num_images, -1])

        if self.loss == "MSE":
            return self.MSE(ref_pixels, align_pixels)
        elif self.loss == "NormCorr":
            return self.norm_corr_loss(ref_pixels, align_pixels)
        else:
            return self.corr_loss(ref_pixels, align_pixels)

    @tf.function
    def apply_patch_transform(self, x_min, y_min, image_patch, num_cut=1000, pool=1):
        """Deprecated"""
        warnings.warn(f"Deprecated. Use `apply_transform` instead.", DeprecationWarning)
        xv = np.arange(x_min, x_min + image_patch.shape[0]) / (pool)
        yv = np.arange(y_min, y_min + image_patch.shape[1]) / (pool)

        pixel_ind = (
            np.concatenate(
                [
                    np.tile(xv[None, :, None], [1, 1, yv.shape[0]]),
                    np.tile(yv[None, None, :], [1, xv.shape[0], 1]),
                ],
                axis=0,
            )
            .reshape([2, -1])
            .transpose()
        )

        pixel_transform = (
            tf.concat(
                [
                    self.forward_pass(
                        split[:, 0],
                        split[:, 1],
                    )[0]
                    for split in np.array_split(pixel_ind, num_cut)
                ],
                axis=0,
            ).numpy()
            * self.offset
        )

        xv = np.arange(image_patch.shape[0])
        yv = np.arange(image_patch.shape[1])

        pixel_ind = np.concatenate(
            [
                np.tile(xv[None, :, None], [1, 1, yv.shape[0]]),
                np.tile(yv[None, None, :], [1, xv.shape[0], 1]),
            ],
            axis=0,
        ).reshape([2, -1])

        new_ind = pixel_ind.transpose() + (pixel_transform) * pool

        new_ind = (
            new_ind.transpose()
            .reshape([2, image_patch.shape[0], image_patch.shape[1]])
            .transpose([1, 2, 0])
        )
        new_ind[new_ind < 0] = 0
        new_ind[:, :, 0][new_ind[:, :, 0] > image_patch.shape[0]] = image_patch.shape[0]
        new_ind[:, :, 1][new_ind[:, :, 1] > image_patch.shape[1]] = image_patch.shape[1]

        return skimage.transform.warp(
            image_patch.astype("float32"),
            np.array([new_ind[:, :, 0], new_ind[:, :, 1]]),
            mode="nearest",
        )

    @tf.function
    def compute_apply_gradients(self, x_ind, y_ind, optimizer):
        with tf.GradientTape() as tape:
            loss = self.compute_loss(x_ind, y_ind)
        gradients = tape.gradient(loss, self.trainable_variables)
        optimizer.apply_gradients(zip(gradients, self.trainable_variables))
        return loss

    # consider adding: @tf.function(jit_compile=True)
    @tf.function
    def train(
        self,
        num_steps=int(np.power(2, 14)),
        batch_size=2048,
        LR=0.0005,
        verbose=4,
        lr__sched=True,
    ):
        """
        Considerations:
        * Does sample_ind do whats required? Isnt it just mapping a diagonal line?
        """
        x_bin = np.where(self.bin_mask == 1)[0].astype("int32")
        y_bin = np.where(self.bin_mask == 1)[1].astype("int32")
        optimizer = tf.keras.optimizers.Adam(learning_rate=LR)

        loss_count = 0
        count = 0

        tq = trange(num_steps, leave=True, desc="")
        for _ in tq:  # range(0, epochs + 1):
            if (lr__sched) and (_ == int(num_steps * 0.75)):
                optimizer.lr.assign(LR * 0.1)

            sample_ind = np.random.randint(
                size=batch_size, low=0, high=x_bin.shape[0], dtype=np.int32
            )
            # tf.random.uniform(shape=[batch_size], maxval=x_bin.shape[0], dtype=tf.int32)

            x_ind = x_bin[sample_ind]
            y_ind = y_bin[sample_ind]

            pixel_glimpses_float, pixel_glimpses_ref = self.get_glimpses(x_ind, y_ind)
            
            x_ind, y_ind = (
                x_ind / self.references.shape[1],
                y_ind / self.references.shape[2],
            )
            with tf.GradientTape() as tape:
                vec, sig = self.forward_pass(x_ind, y_ind)  # vec is pixel after transformation, not the vector
                output_offset = self.vec_to_field(vec, sig)
                loss = self.compute_loss(
                    output_offset, pixel_glimpses_float, pixel_glimpses_ref
                )

            gradients = tape.gradient(loss, self.trainable_variables)
            optimizer.apply_gradients(zip(gradients, self.trainable_variables))

            count += 1
            loss_count += loss.numpy()
            if (_ % verbose == 0) and (verbose > 0):
                tq.set_description("%.5f" % (loss_count / count))
                tq.refresh()  # to show immediately the update

                loss_count = 0
                count = 0

        return 0

    def compute_transform(self, num_cut=5000, pool=None):
        """
        Calculate transformation for each pixel.
        
        Parameters:
        * pool: Pooling factor
        * num_cut: Reduce memory requirements by splitting the pixels into batches
        """
        if pool is None:
            pool = self.pool

        xv = np.arange(self.references.shape[1] * pool) / (
            self.references.shape[1] * pool
        )
        yv = np.arange(self.references.shape[2] * pool) / (
            self.references.shape[2] * pool
        )

        pixel_ind = (
            np.concatenate(
                [
                    np.tile(xv[None, :, None], [1, 1, yv.shape[0]]),
                    np.tile(yv[None, None, :], [1, xv.shape[0], 1]),
                ],
                axis=0,
            )
            .reshape([2, -1])
            .transpose()
        )

        num_cut = min(num_cut, pixel_ind.shape[0])
        pixel_transform = (
            tf.concat(
                [
                    self.forward_pass(
                        split[:, 0],
                        split[:, 1],
                    )[0]
                    for split in np.array_split(pixel_ind, num_cut)
                ],
                axis=0,
            ).numpy()
            * self.offset
        )

        xv = np.arange(self.references.shape[1] * pool)
        yv = np.arange(self.references.shape[2] * pool)

        pixel_ind = np.concatenate(
            [
                np.tile(xv[None, :, None], [1, 1, yv.shape[0]]),
                np.tile(yv[None, None, :], [1, xv.shape[0], 1]),
            ],
            axis=0,
        ).reshape([2, -1])

        new_ind = pixel_ind.transpose() - (pixel_transform) * pool

        new_ind = (
            new_ind.transpose()
            .reshape(
                [
                    2,
                    self.references.shape[1] * pool,
                    self.references.shape[2] * pool,
                ]
            )
            .transpose([1, 2, 0])
        )
        new_ind[new_ind < 0] = 0
        new_ind[:, :, 0][
            new_ind[:, :, 0] > self.references.shape[1] * pool
        ] = (
            self.references.shape[1] * pool
        )
        new_ind[:, :, 1][
            new_ind[:, :, 1] > self.references.shape[2] * pool
        ] = (
            self.references.shape[2] * pool
        )

        self.full_transform = new_ind

        return 0

    def apply_transform(self, image):
        """
        Apply the transformation to an image.
        """
        image_warped = skimage.transform.warp(
            image.astype("float32"),
            np.array([
                self.full_transform[:, :, 0],
                self.full_transform[:, :, 1]
            ]),
            order=0,
        )

        return image_warped
    