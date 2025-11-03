import numpy as np
import torch
import cupy as cp
import warnings
import utils

class MIRAGE(torch.nn.Module):
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
        save_glimpses_file_path=None,
        save_transform_file_path=None,
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
        super(MIRAGE, self).__init__()

        # Set settings
        torch.backends.cudnn.enabled = True

        # Assertions
        # * Images shape must align
        # * Images must be 2D or a list of 2D images
        # * Images must be between 0 and 1
        # * Mask must be same shape as images
        # * GPU must be available (for now)
        assert references.shape == images.shape, (
            f"Images must be same shape. Got {references.shape} (aligning) and "
            f"{images.shape} (reference)."
        )
        assert references.ndim in [
            2,
            3,
        ], f"Images must be 2D or a list of 2D images. Got dimension {references.ndim}."
        assert np.all(references >= 0) and np.all(references <= 1), (
            f"Image values must be between 0 and 1. Got min: {np.min(references)} "
            f"and max: {np.max(references)}. Transform images to 0-1 using for example: `<img> / 255`"
        )
        assert (bin_mask is None) or (bin_mask.shape == references.shape), (
            f"Mask must be same shape as images. Got {bin_mask.shape} (mask) and "
            f"{references.shape} (images)."
        )
        # TODO: CPU cannot support edges of meshes in function gather_nd() --> Only sample from "valid" region
        assert tf.config.list_physical_devices(
            "GPU"
        ), f"GPU must be available. Got {tf.config.list_physical_devices('GPU')}."
        if loss not in ["SSIM"]:
            warnings.warn(
                f"Loss function {loss} is not test. Recommended to use `SSIM` instead."
            )

        # Initialize
        self.save_glimpses_file_path = (
            save_glimpses_file_path  # Rohit added this for debugging
        )
        self.save_transform_file_path = save_transform_file_path

        if save_glimpses_file_path is not None and os.path.exists(
            save_glimpses_file_path
        ):
            os.remove(save_glimpses_file_path)
        if save_transform_file_path is not None and os.path.exists(
            save_transform_file_path
        ):
            os.remove(save_transform_file_path)

        self.num_layers = num_layers
        self.num_neurons = num_neurons
        self.references = references.astype("float32")  # TensorFlow prefers float32
        self.images = images.astype("float32")
        self.bin_mask = bin_mask
        self.loss = loss

        # Added by Rohit to expand dimensions if needed - allows user to pass in 2D images
        if self.references.ndim == 2:
            self.references = np.expand_dims(self.references, axis=0)
        if self.images.ndim == 2:
            self.images = np.expand_dims(self.images, axis=0)
        if self.bin_mask is not None and self.bin_mask.ndim == 2:
            ic("here 2")
            self.bin_mask = np.expand_dims(self.bin_mask, axis=0)
        if self.bin_mask is None:
            ic("here 1")
            self.bin_mask = np.ones(self.references.shape[1:3], dtype=np.uint8)

        # Gets the number of images (for our purposes, we are just using the first channel) - num_images is referring to the number of images
        # images input is (C, H, W)
        self.num_images = self.references.shape[0]
        self.image_height = self.references.shape[
            1
        ]  # All channels should have the same dimensions
        self.image_width = self.references.shape[2]

        ic(self.references.shape)
        ic(self.images.shape)
        ic(self.bin_mask.shape)
        ic(pool)
        ic(self.image_height)
        ic(self.image_width)
        ic(self.references.astype("float")[:, :, :, None].shape)

        self.pos_encoding_L = 10
        self.ssim_map = tf.convert_to_tensor(
            get_ssim_map(self.references[0], self.images[0])
        )

        # Pooling to reduce image size
        if (pool is not None) and (pool > 1):
            self.pool = pool

            # This snippet downsamples.a grayscale imageusing average pooling
            # 1. Converts to a tensor
            # 2. Adds a new dimension of size 1 at the end of the array, then pools on that tensor
            # (TF expects a tensor of rank N+2, of shape [batch_size + input_spatial_shape + [num_channels]])
            # So for our input 2D images, it needs a rank of N + 2 = 4
            # So expects a shape of [batch_size, height, width, channels] -> batch_size and channels are both 1
            # 3. Runs average pooling and then gets it back to [batch_size, height, width]
            with tf.device("/CPU:0"):
                self.references = (
                    tf.squeeze(  # Remove all size 1 dimensions - so if there's only one channel i.e. (1, 500, 500) -> will become (500, 500)
                        (
                            tf.nn.avg_pool(  # Uses mean pooling based on self.pool
                                self.references.astype("float")[:, :, :, None],
                                ksize=[
                                    self.pool,
                                    self.pool,
                                ],  # not sure if this works - ValueError: ksize should be of length 1, 3 or 5. Received: ksize=[3, 3] of length 2 is i pass in self.pool = 3
                                strides=[
                                    self.pool,
                                    self.pool,
                                ],  # ksize determines window, strides determine how far the window moves after each calculation
                                padding="SAME",  # VALID is no padding, SAME ensures that ouptut size matches input size divided by stride (makes sense because image_height and image_width have to remain the same)
                                # pooling probably speeds up computation (but not sure if this is necessary? could just take alignment at highest resolution and downsample manually?)
                            )
                        ),
                        axis=-1,  # removes the last dimsension of the tensor - so will remove the None slice that we added earlier
                    )
                    .numpy()
                    .astype("float32")
                )
                # Same thing here as for references
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
                # Same thing here as for references
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
            # Default is no pooling
            self.pool = 1

        # Brings offset and pad down based on pooling factor - again why is pooling necessary? is it to smooth things out?
        self.offset = int(offset / self.pool)
        self.pad = int(pad / self.pool)

        # Gives weight to specific images? Not entirely sure why
        if coeff is None:
            self.coeff = np.ones(self.num_images)
        else:
            self.coeff = np.asarray(coeff)

        # Initalizes NN layers
        self.layers_custom = []
        # Sets random weights to each of the layers - Glorot Normal draws weights from a normal distribution
        # Variance of the distribution is scaled based on the number of input and output neurons of the layer, meaning that the variance of the acitvations
        # and backprop gradients (signals don't grow or shrink too quickly)
        # Var = 2 / (num_neurons_in + num_neurons_out) -> apparently goes well with sigmoid and tanh
        self.initializer = tf.keras.initializers.GlorotNormal()

        # Runs through layers
        for i in range(self.num_layers):
            # appends each layer to the list of layers
            self.layers_custom.append(
                # Creates a dense layer -> fully connected layer to approximate a continuous function
                # We choose a dense layer here because we're not trying to learn image features but a non_linear mapping from coordinates to a transformation vector
                # Dense layers are flexible
                tf.keras.layers.Dense(
                    units=self.num_neurons,
                    kernel_initializer=self.initializer,
                    bias_initializer=self.initializer,
                )
            )

        # These are the final two layer
        # Vec Layer are the x, y shifts that are determined by the NN
        # We set them to 0 because we start with no shift at all
        self.vec_layer = tf.keras.layers.Dense(
            units=2,
            kernel_initializer=tf.keras.initializers.Zeros(),
            bias_initializer=tf.keras.initializers.Zeros(),
        )
        # This provides a confidence score - again we are not confident at all about anything so we set them to zeroes initially
        # We only need units = 1 because this corresponds to each vector in vec_layer
        self.sig_layer = tf.keras.layers.Dense(
            units=1,
            kernel_initializer=tf.keras.initializers.Zeros(),
            bias_initializer=tf.keras.initializers.Zeros(),
        )

        # # These are the final two layer
        # # Vec Layer are the x, y shifts that are determined by the NN
        # # We set them to 0 because we start with no shift at all
        # self.vec_layer = tf.keras.layers.Dense(
        #     units=2,
        #     kernel_initializer=self.initializer,
        #     bias_initializer=self.initializer,
        # )
        # # This provides a confidence score - again we are not confident at all about anything so we set them to zeroes initially
        # # We only need units = 1 because this corresponds to each vector in vec_layer
        # self.sig_layer = tf.keras.layers.Dense(
        #     units=1,
        #     kernel_initializer=self.initializer,
        #     bias_initializer=self.initializer,
        # )

        # Gives us a grid of coordinates
        # tf.range creates a 1D tensor of images -  we start with offset - pad (in the basic case these are both 12 and 12)
        # So in total we end up with a 49 x 49 square (x and y)
        self.y_mesh, self.x_mesh = tf.meshgrid(
            tf.range(-self.offset - self.pad, self.offset + self.pad + 1),
            tf.range(-self.offset - self.pad, self.offset + self.pad + 1),
        )
        # ic(self.y_mesh, self.x_mesh)
        # ic(self.y_mesh.shape, self.x_mesh.shape)

        initial_scale = 3.0
        self.sig_scale_factor = tf.Variable(
            initial_value=initial_scale, trainable=True, name="sig_scale_factor"
        )

        # Create a simple dissimilarity map
        # Blurring helps capture regional differences rather than just pixel noise
        ref_blur = skimage.filters.gaussian(self.references[0], sigma=5)
        img_blur = skimage.filters.gaussian(self.images[0], sigma=5)
        dissimilarity_map = np.abs(ref_blur - img_blur)

        # Normalize to [0, 1] and store it
        self.dissimilarity_map = (dissimilarity_map / dissimilarity_map.max()).astype(
            "float32"
        )

        self.edge_mask = (
            skimage.filters.sobel(self.references[0])
            > np.percentile(skimage.filters.sobel(self.references[0]), 75)
        ).astype("float32")

        """
        ic| self.y_mesh: <tf.Tensor: shape=(49, 49), dtype=int32, numpy=
                 array([[-24, -23, -22, ...,  22,  23,  24],
                        [-24, -23, -22, ...,  22,  23,  24],
                        [-24, -23, -22, ...,  22,  23,  24],
                        ...,
                        [-24, -23, -22, ...,  22,  23,  24],
                        [-24, -23, -22, ...,  22,  23,  24],
                                [-24, -23, -22, ...,  22,  23,  24]], dtype=int32)>
            self.x_mesh: <tf.Tensor: shape=(49, 49), dtype=int32, numpy=
                        array([[-24, -24, -24, ..., -24, -24, -24],
                                [-23, -23, -23, ..., -23, -23, -23],
                                [-22, -22, -22, ..., -22, -22, -22],
                                ...,
                                [ 22,  22,  22, ...,  22,  22,  22],
                                [ 23,  23,  23, ...,  23,  23,  23],
                                [ 24,  24,  24, ...,  24,  24,  24]], dtype=int32)>
        ic| self.y_mesh.shape: TensorShape([49, 49])
            self.x_mesh.shape: TensorShape([49, 49])
        
        """

    def get_glimpses(self, x_ind, y_ind):
        # ... (Original logic remains here) ...
        # (Original logic for get_glimpses remains here)
        X = (
            tf.tile(self.x_mesh[None, :, :], [x_ind.shape[0], 1, 1])
            + x_ind[:, None, None]
        )
        Y = (
            tf.tile(self.y_mesh[None, :, :], [y_ind.shape[0], 1, 1])
            + y_ind[:, None, None]
        )
        XY = tf.stack((X, Y), axis=-1)

        if self.save_glimpses_file_path is not None:
            starting_coords = tf.stack((x_ind, y_ind), axis=1)
            # Assuming append_to_txt_file is defined elsewhere or this line is removed for non-essential code
            # self.append_to_txt_file(self.save_glimpses_file_path, starting_coords.numpy())
            pass

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

        XY = XY[:, self.offset : -self.offset, self.offset : -self.offset, :]

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

    def train(
        self,
        num_steps=int(np.power(2, 14)),
        batch_size=2048,
        LR=0.0005,
        verbose=4,
        lr__sched=True,
    ):
        """
        ... (Docstring truncated)
        """
        x_bin = np.where(self.bin_mask == 1)[0].astype("int32")
        y_bin = np.where(self.bin_mask == 1)[1].astype("int32")

        optimizer = tf.keras.optimizers.Adam(learning_rate=LR)

        loss_count = 0
        count = 0

        tq = trange(num_steps, leave=True, desc="")
        for _ in tq:
            if (lr__sched) and (_ == int(num_steps * 0.75)):
                optimizer.lr.assign(LR * 0.1)

            high = min(x_bin.shape[0], 2**31 - 1)
            sample_ind = np.random.randint(
                size=batch_size, low=0, high=high, dtype=np.int32
            )

            # Get image-space indices for the center of the glimpse
            x_ind_img = x_bin[sample_ind]
            y_ind_img = y_bin[sample_ind]

            # Get feature inputs at these locations
            dissimilarity_vals = self.dissimilarity_map[x_ind_img, y_ind_img]
            edge_vals = self.edge_mask[x_ind_img, y_ind_img]  # New edge input

            pixel_glimpses_float, pixel_glimpses_ref = self.get_glimpses(
                x_ind_img, y_ind_img
            )

            # Normalize the indices to be between 0 and 1
            x_ind_norm = x_ind_img / self.references.shape[1]
            y_ind_norm = y_ind_img / self.references.shape[2]

            with tf.GradientTape() as tape:
                vec, sig = self.forward_pass(
                    x_ind_norm, y_ind_norm, dissimilarity_vals, edge_vals
                )  # Updated forward_pass call
                output_offset = self.vec_to_field(vec, sig)
                loss = self.compute_loss(
                    output_offset, pixel_glimpses_float, pixel_glimpses_ref
                )

            gradients = tape.gradient(loss, self.trainable_variables)
            optimizer.apply_gradients(zip(gradients, self.trainable_variables))

            # --- PLOTTING BLOCK STARTS HERE ---
            if _ > 0 and _ % 200 == 0:
                # Create the magnitude heatmap
                magnitude_map = self.create_magnitude_heatmap()

                fig, axes = plt.subplots(1, 2, figsize=(12, 6))
                
                # Show the reference image
                axes[0].imshow(self.images[0], cmap='gray')
                axes[0].set_title("Reference Image")
                axes[0].axis('off')

                # Show the magnitude heatmap
                im = axes[1].imshow(magnitude_map, cmap='inferno') # 'inferno' is a good heatmap colormap
                axes[1].set_title(f"Displacement Magnitude (Step {_})")
                axes[1].axis('off')
                plt.colorbar(im, ax=axes[1], label="Shift Magnitude (pixels)")
                
                # Add overall title and save
                loss_val = loss.numpy()
                fig.suptitle(f"Step {_}, Loss: {loss_val:.5f}")
                plt.tight_layout()
                
                save_path = f"./deformation_images/heatmap_step_{_:05d}.png"
                os.makedirs(os.path.dirname(save_path), exist_ok=True)
                plt.savefig(save_path)
                plt.close(fig)
            # --- PLOTTING BLOCK ENDS HERE --

            count += 1
            loss_count += loss.numpy()
            if (_ % verbose == 0) and (verbose > 0):
                tq.set_description("%.5f" % (loss_count / count))
                tq.refresh()

                loss_count = 0
                count = 0

        return 0

    def create_magnitude_heatmap(self):
        """
        Computes the dense transformation field and returns a 2D heatmap
        of the displacement magnitude for each pixel.
        """
        h, w = self.references.shape[1], self.references.shape[2]
        y_coords, x_coords = np.meshgrid(np.arange(h), np.arange(w), indexing="ij")

        y_norm = y_coords.flatten().astype("float32") / h
        x_norm = x_coords.flatten().astype("float32") / w

        # Predict the dense vector field in batches
        batch_size = 4096
        all_vecs = []
        for i in trange(
            0, len(x_norm), batch_size, leave=False, desc="Generating heatmap"
        ):
            vec_batch, _ = self.forward_pass(
                tf.constant(y_norm[i : i + batch_size]),
                tf.constant(x_norm[i : i + batch_size]),
                tf.constant(
                    self.dissimilarity_map[
                        y_coords.flatten()[i : i + batch_size],
                        x_coords.flatten()[i : i + batch_size],
                    ]
                ),
            )
            all_vecs.append(vec_batch.numpy())

        # Calculate pixel shift magnitude
        vec_field = np.concatenate(all_vecs, axis=0)
        magnitude = np.linalg.norm(vec_field, axis=-1) * self.offset

        return magnitude.reshape(h, w)

    # Then modify forward_pass to use positional encoding:
    @tf.function
    def forward_pass(self, x_ind_norm, y_ind_norm, dissimilarity_input, edge_input):
        # Normalize to [-1, 1] range
        coords_norm = tf.stack(
            [
                2 * x_ind_norm - 1,
                2 * y_ind_norm - 1,
            ],
            axis=1,
        )

        # 1. Positional Encoding
        pos_encoded = get_positional_encoding(coords_norm, L=self.pos_encoding_L)

        # 2. Concatenate with learned features and static image features
        output = tf.concat(
            [
                tf.cast(pos_encoded, tf.float32),
                tf.cast(coords_norm, tf.float32),  # Also keep raw coordinates
                tf.cast(dissimilarity_input[:, None], tf.float32),
                tf.cast(edge_input[:, None], tf.float32),  # New edge input
            ],
            axis=1,
        )

        for i in range(self.num_layers):
            output = tf.nn.silu(self.layers_custom[i](output))

        vec_output = tf.nn.tanh(self.vec_layer(output))
        # Use tf.exp(log_sig_scale_factor) for positive scale factor
        sig_output = tf.nn.softplus(self.sig_layer(output)) * (
            tf.exp(self.sig_scale_factor) + 1e-6
        )

        return (vec_output, sig_output)

    @tf.function
    def vec_to_field(self, vec, sig):
        # ic(vec.shape)
        # ic(sig.shape)
        # vec has shape (512, 2)
        # sig has shape(512, 1)
        # ic(sig)

        # for each glimpse, gets an x, y transformation from vec and a corresponding sig that is a confidence score
        x_dist = (
            # Squaring ensures all distances are positive and gives greater weight to larger differences
            # we just want the magnitude, the direction does not matter
            # we want to penalize deviation, so squaring brings the distance down
            tf.square(
                # linspace creates a 1d tensor from -1 to 1 with evenly spaced numbers that correspond to the number of offset pixels
                # in both directions
                # this is then casted so that theres a new dimention -> leading to a (1, <number_of_offset_pixels>)
                tf.cast(tf.linspace(-1, 1, 2 * self.offset + 1), dtype=tf.float32)[
                    None, :
                ]
                # this is X so then we subtract the vec x coordinate to get the distance between original and new coordinates
                # we end up with a grid in the [-1, 1] range and each column is the predicted x_shift
                - vec[:, 0][:, None]
            )
            / sig
        )

        # same thing as for x_dist
        y_dist = (
            tf.square(
                tf.cast(tf.linspace(-1, 1, 2 * self.offset + 1), dtype=tf.float32)[
                    None, :
                ]
                - vec[:, 1][:, None]
            )
            / sig
        )

        # both dists becomes a tensor of size (512, 25, 1) + [512, 25, 1] so they can be added
        # For each glimpse we have a [25, 25] grid that represents the combined squared distance from the original position
        field = x_dist[:, :, None] + y_dist[:, None, :]

        # plt.imshow(field[0])
        # plt.show()
        # plt.title("Field before softmax")
        # plt.savefig("field_before_softmax.png")
        # final reshape takes the 1D tensor of distances and brings it back to (512, 25, 25)
        # these 2d slices now have values between 0 and 1 that represent the shift values
        field = tf.reshape(
            # inner reshape takes the field (512, 25, 25) and flattens it -> new shape is (512, 25 * 25)
            # we then negate that so that smaller distances become a larger magnitude and further distances become smaller in
            # this gets passed into the softmax function whihc converts these values in a probability distribtution - all output values for each glimpse sum to 1
            # the most probable transofmation will be the one that minimizes the shift while shifts that dramatically shift the pixels will be penalized
            tf.nn.softmax(-tf.reshape(field, [vec.shape[0], -1]), axis=-1),
            [vec.shape[0], 2 * self.offset + 1, 2 * self.offset + 1],
        )
        # plt.imshow(field[0])
        # plt.show()

        # plt.title("Field after softmax")
        # plt.savefig("field_after_softmax.png")

        return field

    @tf.function
    def compute_loss(self, output_offset, pixel_glimpses_float, pixel_glimpses_ref):
        # ... (Convolution logic remains here) ...
        pixel_glimpses_float = tf.nn.depthwise_conv2d(
            pixel_glimpses_float,
            tf.transpose(output_offset, [1, 2, 0])[:, :, :, None],
            strides=[1, 1, 1, 1],
            padding="VALID",
        )
        # pixel_glimpses_float shape is (1, 25, 25, 512)

        if self.loss == "MultiSSIM":
            # Assuming multi_ssim is implemented with DSSIM logic
            return self.multi_ssim(pixel_glimpses_float, pixel_glimpses_ref)

        if self.loss == "SSIM":
            return self.SSIM(pixel_glimpses_float, pixel_glimpses_ref)

        if self.loss == "DSSIM":
            return self.DSSIM(pixel_glimpses_float, pixel_glimpses_ref)

        align_pixels = tf.reshape(pixel_glimpses_float, [self.num_images, -1])
        ref_pixels = tf.reshape(pixel_glimpses_ref, [self.num_images, -1])

        if self.loss == "MSE":
            return self.MSE(ref_pixels, align_pixels)
        elif self.loss == "NormCorr":
            return self.norm_corr_loss(ref_pixels, align_pixels)
        else:
            return self.corr_loss(ref_pixels, align_pixels)

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

    def compute_transform(self, num_cut=5000, pool=None):
        """
        Calculate transformation for each pixel in memory-safe batches.

        Parameters:
        * pool: Pooling factor
        * num_cut: Batch size (number of pixels to process at once)
        """
        if pool is None:
            pool = self.pool

        h, w = self.references.shape[1], self.references.shape[2]

        # Raw pixel coordinates
        xv = np.arange(h * pool)
        yv = np.arange(w * pool)
        pixel_ind = np.stack(np.meshgrid(xv, yv, indexing="ij"), axis=-1).reshape(-1, 2)

        # Normalized coords
        x_norm_all = pixel_ind[:, 0] / float(h * pool)
        y_norm_all = pixel_ind[:, 1] / float(w * pool)

        # Dissimilarity values
        dissim_all = self.dissimilarity_map[
            pixel_ind[:, 0] // pool, pixel_ind[:, 1] // pool
        ]

        # Edge values
        edge_vals = self.edge_mask[pixel_ind[:, 0] // pool, pixel_ind[:, 1] // pool]

        # Storage for dense transform
        pixel_transform = np.zeros((pixel_ind.shape[0], 2), dtype=np.float32)

        # Process in batches
        for start in range(0, pixel_ind.shape[0], num_cut):
            end = min(start + num_cut, pixel_ind.shape[0])

            coords_x = tf.constant(x_norm_all[start:end], dtype=tf.float32)
            coords_y = tf.constant(y_norm_all[start:end], dtype=tf.float32)
            dissim = tf.constant(dissim_all[start:end], dtype=tf.float32)
            edge_input = tf.constant(edge_vals[start:end], dtype=tf.float32)

            vec_batch, _ = self.forward_pass(coords_x, coords_y, dissim, edge_input)
            pixel_transform[start:end] = vec_batch.numpy() * self.offset

        # Apply displacements to pixel coordinates
        new_ind = pixel_ind.astype(np.float32) - (pixel_transform * pool)

        # Reshape to full image grid
        new_ind = new_ind.reshape(h * pool, w * pool, 2)

        # Clamp to valid range
        new_ind[new_ind < 0] = 0
        new_ind[:, :, 0][new_ind[:, :, 0] > h * pool - 1] = h * pool - 1
        new_ind[:, :, 1][new_ind[:, :, 1] > w * pool - 1] = w * pool - 1

        # Store dense transform (compatible with apply_transform)
        self.full_transform = new_ind

        if self.save_transform_file_path:
            np.save(self.save_transform_file_path, self.full_transform)

        return 0

    def get_positional_encoding(self, coords, L=10):
        """
        Apply Positional Encoding to coordinates (x, y).

        Input coords: Tensor of shape (N, 2) normalized to [-1, 1], dtype=torch.float32.
        Output: Tensor of shape (N, 4*L).
        """
        # 1. Create the base frequencies tensor: [2^0, 2^1, ..., 2^(L-1)]
        # Use tf.pow for TensorFlow-native exponentiation.
        base_powers = torch.pow(2.0, torch.arange(L, dtype=torch.float32))

        # 2. Multiply by pi, ensuring pi is a tf.float32 constant
        freqs = base_powers * torch.tensor(np.pi, dtype=torch.float32)

        # Outer product: (N, 2) x (1, L) -> (N, 2, L)
        # This is where the multiplication error was occurring.
        sincos_input = torch.tensor(coords[..., None], dtype=torch.float32) * torch.tensor(freqs[None, None, :], dtype=torch.float32)
        

        # Shape (N, 2, L)
        sin_part = torch.sin(sincos_input)
        cos_part = torch.cos(sincos_input)

        # Concatenate sin and cos components and flatten the last dimension.
        # Output shape: (N, 4*L)
        output = torch.cat([sin_part, cos_part], dim=-1)
        return output.reshape((coords.shape[0], -1))

    def apply_transform(self, image):
        """
        Apply the transformation to an image.
        """
        image_warped = skimage.transform.warp(
            image.astype("float32"),
            np.array([self.full_transform[:, :, 0], self.full_transform[:, :, 1]]),
            order=0,
        )

        return image_warped
