"""
MIRAGE: Multi-modal Image Registration via Adaptive Glimpse Encoding

Production-ready implementation of the MIRAGE image registration algorithm.
Aligns a moving image to a fixed reference image using a coordinate-based
neural network that predicts per-pixel displacement fields.
"""

import os
import time
import warnings

import cv2
import numpy as np
import skimage.transform
import tensorflow as tf
from tqdm import trange


os.environ["TF_FORCE_GPU_ALLOW_GROWTH"] = "true"


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

def create_rgb_overlay(fixed: np.ndarray, moving: np.ndarray) -> np.ndarray:
    """
    Create a red-cyan RGB overlay for visualising alignment quality.

    Red channel  = fixed image.
    Green + Blue = moving image (appears cyan).
    White pixels indicate well-aligned regions.
    """
    fixed = fixed.astype(np.float32)
    moving = moving.astype(np.float32)

    if fixed.max() > 0:
        fixed /= fixed.max()
    if moving.max() > 0:
        moving /= moving.max()

    rgb = np.zeros((*fixed.shape, 3), dtype=np.float32)
    rgb[..., 0] = fixed
    rgb[..., 1] = moving
    rgb[..., 2] = moving
    return np.clip(rgb, 0, 1)


@tf.function
def get_positional_encoding(coords: tf.Tensor, L: int = 10) -> tf.Tensor:
    """
    Apply sinusoidal positional encoding to 2-D coordinates.

    Parameters
    ----------
    coords : tf.Tensor, shape (N, 2), dtype float32
        Normalised coordinates in [-1, 1].
    L : int
        Number of frequency octaves.

    Returns
    -------
    tf.Tensor, shape (N, 4*L)
    """
    base_powers = tf.pow(
        tf.constant(2, dtype=tf.float32), tf.range(L, dtype=tf.float32)
    )
    freqs = base_powers * tf.constant(np.pi, dtype=tf.float32)

    # (N, 2, L)
    sincos_input = (
        tf.cast(coords[..., None], tf.float32)
        * tf.cast(freqs[None, None, :], tf.float32)
    )

    sin_part = tf.sin(sincos_input)
    cos_part = tf.cos(sincos_input)

    encoded = tf.concat([sin_part, cos_part], axis=-1)
    return tf.reshape(encoded, (tf.shape(coords)[0], -1))


# ---------------------------------------------------------------------------
# Main model
# ---------------------------------------------------------------------------

class MIRAGE(tf.keras.Model):
    """
    MIRAGE image registration model.

    Learns a continuous, coordinate-based displacement field that maps pixels
    in a moving image onto a fixed reference image.

    Parameters
    ----------
    references : np.ndarray
        Fixed reference image(s). Shape (H, W) or (C, H, W). Values in [0, 1].
    images : np.ndarray
        Moving image(s) to be aligned. Same shape as ``references``.
    bin_mask : np.ndarray, optional
        Binary mask (1 = valid pixel). Shape (H, W) or (C, H, W).
        Defaults to all-ones (all pixels used).
    num_layers : int
        Number of hidden layers in the MLP.
    num_neurons : int
        Width of each hidden layer.
    pad : int
        Half-size of the reference glimpse (in pixels, pre-pooling).
    offset : int
        Maximum displacement magnitude (in pixels, pre-pooling).
    batch_size : int
        Number of glimpse centres sampled per training step.
    LR : float
        Initial learning rate for AdamW.
    LR_sched : bool
        If True, reduce LR to 10 % at 75 % of training.
    pool : int, optional
        Average-pooling factor applied to images before training (speeds up
        computation; transforms are still returned at full resolution).
    loss : str
        Loss function. Only ``"SSIM"`` is tested.
    coeff : array-like, optional
        Per-channel weighting coefficients. Defaults to ones.
    save_glimpses_file_path : str, optional
        If set, save glimpse debug data to this path.
    save_transform_file_path : str, optional
        If set, save the computed transform array to this path (.npy).
    """

    def __init__(
        self,
        references: np.ndarray,
        images: np.ndarray,
        bin_mask: np.ndarray = None,
        num_layers: int = 3,
        num_neurons: int = 1024,
        pad: int = 13,
        offset: int = 15,
        batch_size: int = 512,
        LR: float = 0.001,
        LR_sched: bool = True,
        pool: int = None,
        loss_type: str = "SSIM",
        coeff=None,
        smoothness_weight: float = 0.1,
        smoothness_radius: int = 30,
        rigidity_weight: float = 0.1,
        pos_encoding_L: int = 6,
        dissim_sigma: int = None,
        save_glimpses_file_path: str = None,
        save_transform_file_path: str = None,
    ):
        super().__init__()

        tf.config.run_functions_eagerly(False)

        # ------------------------------------------------------------------
        # Input validation
        # ------------------------------------------------------------------
        assert references.shape == images.shape, (
            f"Images must be the same shape. "
            f"Got references={references.shape}, images={images.shape}."
        )
        assert references.ndim in (2, 3), (
            f"Images must be 2-D or a stack of 2-D images (3-D). "
            f"Got ndim={references.ndim}."
        )
        assert np.all(references >= 0) and np.all(references <= 1), (
            f"Image values must be in [0, 1]. "
            f"Got min={np.min(references):.4f}, max={np.max(references):.4f}. "
            f"Normalise with e.g. `img / 255`."
        )
        if bin_mask is not None:
            assert bin_mask.shape == references.shape, (
                f"bin_mask must have the same shape as the images. "
                f"Got mask={bin_mask.shape}, images={references.shape}."
            )
        assert tf.config.list_physical_devices("GPU"), (
            "A CUDA-capable GPU is required. "
            f"Detected: {tf.config.list_physical_devices('GPU')}."
        )
        if loss_type not in ("SSIM",):
            warnings.warn(
                f"Loss '{loss_type}' is untested. 'SSIM' is recommended.",
                UserWarning,
                stacklevel=2,
            )

        # ------------------------------------------------------------------
        # File paths
        # ------------------------------------------------------------------
        self.save_glimpses_file_path = save_glimpses_file_path
        self.save_transform_file_path = save_transform_file_path

        for path in (save_glimpses_file_path, save_transform_file_path):
            if path is not None and os.path.exists(path):
                os.remove(path)

        # ------------------------------------------------------------------
        # Store settings
        # ------------------------------------------------------------------
        self.num_layers = num_layers
        self.num_neurons = num_neurons
        self.loss_type = loss_type
        self.batch_size = batch_size
        self.LR = LR
        self.LR_sched = LR_sched
        self.pos_encoding_L = pos_encoding_L
        self.smoothness_weight = smoothness_weight
        self.smoothness_radius = smoothness_radius
        self.rigidity_weight = rigidity_weight
        self.num_images = 1 

        # ------------------------------------------------------------------
        # Expand dims: work internally as (C, H, W)
        # ------------------------------------------------------------------
        refs = references.astype("float32")
        imgs = images.astype("float32")

        if refs.ndim == 2:
            refs = np.expand_dims(refs, axis=0)
        if imgs.ndim == 2:
            imgs = np.expand_dims(imgs, axis=0)

        mask = bin_mask
        if mask is not None and mask.ndim == 2:
            mask = np.expand_dims(mask, axis=0)

        # ------------------------------------------------------------------
        # Optional average pooling to reduce resolution during training
        # ------------------------------------------------------------------
        if pool is not None and pool > 1:
            self.pool = pool
            with tf.device("/CPU:0"):
                refs = self._avg_pool(refs)
                imgs = self._avg_pool(imgs)
                if mask is not None:
                    mask = self._avg_pool_mask(mask)

            if mask is None:
                mask = np.ones(refs.shape[1:3], dtype=np.uint8)
        else:
            self.pool = 1
            if mask is None:
                mask = np.ones(refs.shape[1:3], dtype=np.uint8)

        self.references = refs
        self.images = imgs
        self.bin_mask = mask

        # ------------------------------------------------------------------
        # Scale offset and pad to pooled resolution
        # ------------------------------------------------------------------
        self.offset = int(offset / self.pool)
        self.pad = int(pad / self.pool)

        self.image_height = self.references.shape[1]
        self.image_width = self.references.shape[2]

        # ------------------------------------------------------------------
        # Per-channel coefficients
        # ------------------------------------------------------------------
        self.coeff = (
            np.ones(self.num_images) if coeff is None else np.asarray(coeff)
        )

        # ------------------------------------------------------------------
        # MLP layers
        # ------------------------------------------------------------------
        self.initializer = tf.keras.initializers.GlorotNormal()
        self.layers_custom = [
            tf.keras.layers.Dense(
                units=self.num_neurons,
                kernel_initializer=self.initializer,
                bias_initializer=self.initializer,
            )
            for _ in range(self.num_layers)
        ]

        # Output heads — initialised to zero so the model starts with no shift
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

        # Learnable scale factor for the confidence (sigma) head
        self.log_sig_scale_factor = tf.Variable(
            initial_value=np.log(3.0),
            trainable=True,
            name="log_sig_scale_factor",
            dtype=tf.float32,
        )

        # ------------------------------------------------------------------
        # Glimpse offset meshes
        #
        # NOTE on naming convention used throughout this class:
        #   row_mesh  — integer offsets along the HEIGHT axis (axis-0 in arrays)
        #   col_mesh  — integer offsets along the WIDTH  axis (axis-1 in arrays)
        #
        # np.meshgrid(a, b) returns (grid_of_a_varying_along_cols,
        #                            grid_of_b_varying_along_rows).
        # We therefore call it as meshgrid(col_range, row_range) and unpack
        # as (col_mesh, row_mesh).
        # ------------------------------------------------------------------
        span = np.arange(-self.offset - self.pad, self.offset + self.pad + 1)
        self.col_mesh, self.row_mesh = np.meshgrid(span, span)

        # ------------------------------------------------------------------
        # Dissimilarity map (blurred absolute difference) used as a spatial
        # feature fed to the network alongside positional encoding.
        # ------------------------------------------------------------------
        sigma = dissim_sigma if dissim_sigma is not None else max(offset, 5)

        ref_blur = cv2.GaussianBlur(
            self.references[0].astype(np.float32),
            ksize=(0, 0),
            sigmaX=sigma,
            sigmaY=sigma,
        )
        img_blur = cv2.GaussianBlur(
            self.images[0].astype(np.float32),
            ksize=(0, 0),
            sigmaX=sigma,
            sigmaY=sigma,
        )
        diff = np.abs(ref_blur - img_blur)
        dmax = diff.max()
        self.dissimilarity_map = (diff / dmax if dmax > 0 else diff).astype(
            "float32"
        )

    # -----------------------------------------------------------------------
    # Private helpers
    # -----------------------------------------------------------------------

    def _avg_pool(self, arr: np.ndarray) -> np.ndarray:
        """Average-pool a (C, H, W) float32 array by self.pool. Returns float32."""
        return (
            tf.squeeze(
                tf.nn.avg_pool(
                    arr.astype("float64")[:, :, :, None],
                    ksize=[self.pool, self.pool],
                    strides=[self.pool, self.pool],
                    padding="SAME",
                ),
                axis=-1,
            )
            .numpy()
            .astype("float32")
        )

    def _avg_pool_mask(self, mask: np.ndarray) -> np.ndarray:
        """Average-pool a (H, W) binary mask by self.pool. Returns int32 (threshold 0.5)."""
        pooled = (
            tf.squeeze(
                tf.nn.avg_pool(
                    np.squeeze(mask).astype("float64")[None, :, :, None],
                    ksize=[self.pool, self.pool],
                    strides=[self.pool, self.pool],
                    padding="SAME",
                )
            )
            .numpy()
        )
        return (pooled >= 0.5).astype("int32")

    # -----------------------------------------------------------------------
    # Glimpse extraction
    # -----------------------------------------------------------------------

    def get_glimpses(
        self, row_ind: np.ndarray, col_ind: np.ndarray
    ):
        """
        Extract local image patches (glimpses) centred at given pixel locations.

        Parameters
        ----------
        row_ind : np.ndarray, shape (B,)
            Row (height-axis) indices of glimpse centres.
        col_ind : np.ndarray, shape (B,)
            Column (width-axis) indices of glimpse centres.

        Returns
        -------
        pixel_glimpses_moving : np.ndarray, shape (num_images, H_full, W_full, B)
            Glimpses from the moving image (full size, including offset region).
        pixel_glimpses_fixed : np.ndarray, shape (num_images, H_pad, W_pad, B)
            Glimpses from the fixed/reference image (padded region only).
        """
        # Absolute row/col coordinates for every position in the glimpse grid
        # Shape: (B, 2*(offset+pad)+1, 2*(offset+pad)+1)
        rows = self.row_mesh[np.newaxis, :, :] + row_ind[:, np.newaxis, np.newaxis]
        cols = self.col_mesh[np.newaxis, :, :] + col_ind[:, np.newaxis, np.newaxis]

        # In-bounds masks
        rows_ok = (rows >= 0) & (rows < self.image_height)
        cols_ok = (cols >= 0) & (cols < self.image_width)
        in_bounds = rows_ok & cols_ok

        # Clipped (safe) indices for array lookup
        rows_safe = np.clip(rows, 0, self.image_height - 1).astype(np.int64)
        cols_safe = np.clip(cols, 0, self.image_width - 1).astype(np.int64)

        # --- Moving image glimpses (full extent) ---
        glimpses_moving = []
        for c in range(self.num_images):
            vals = self.references[c][rows_safe, cols_safe]
            glimpses_moving.append(np.where(in_bounds, vals, 0))

        # Shape: (num_images, 2*(offset+pad)+1, 2*(offset+pad)+1, B)
        pixel_glimpses_moving = np.transpose(
            np.stack(glimpses_moving, axis=-1),
            axes=[3, 1, 2, 0],
        )

        # --- Fixed/reference image glimpses (padded region only) ---
        rows_off = rows[:, self.offset : -self.offset, self.offset : -self.offset]
        cols_off = cols[:, self.offset : -self.offset, self.offset : -self.offset]

        rows_off_ok = (rows_off >= 0) & (rows_off < self.image_height)
        cols_off_ok = (cols_off >= 0) & (cols_off < self.image_width)
        in_bounds_off = rows_off_ok & cols_off_ok

        rows_off_safe = np.clip(rows_off, 0, self.image_height - 1).astype(np.int64)
        cols_off_safe = np.clip(cols_off, 0, self.image_width - 1).astype(np.int64)

        glimpses_fixed = []
        for c in range(self.num_images):
            vals = self.images[c][rows_off_safe, cols_off_safe]
            glimpses_fixed.append(np.where(in_bounds_off, vals, 0))

        pixel_glimpses_fixed = np.transpose(
            np.stack(glimpses_fixed, axis=-1),
            axes=[3, 1, 2, 0],
        )

        return pixel_glimpses_moving, pixel_glimpses_fixed

    # -----------------------------------------------------------------------
    # Training
    # -----------------------------------------------------------------------


    def train(self, num_steps: int = int(np.power(2, 14)), verbose: int = 4,
              early_stop_patience: int = 500):
        """
        Train the MIRAGE displacement network.

        Parameters
        ----------
        num_steps : int
            Total number of gradient steps.
        verbose : int
            Log loss every ``verbose`` steps. Set to 0 to disable.
        early_stop_patience : int
            If > 0, stop training when the SSIM loss hasn't improved for this
            many steps. The best weights (by SSIM loss) are restored at the end.
            Set to 0 to disable early stopping.
        """
        row_bin = np.where(self.bin_mask == 1)[0].astype("int32")
        col_bin = np.where(self.bin_mask == 1)[1].astype("int32")

        optimizer = tf.keras.optimizers.AdamW(
            learning_rate=self.LR, weight_decay=1e-4
        )

        loss_accum = 0.0
        step_count = 0

        # Early stopping state
        best_ssim_loss = float("inf")
        best_weights = None
        steps_without_improvement = 0
        # Use a running window to smooth out noise
        ssim_window = []
        window_size = max(verbose, 16)

        tq = trange(num_steps, leave=True, desc="")
        for step in tq:
            if self.LR_sched and step == int(num_steps * 0.75):
                optimizer.learning_rate.assign(self.LR * 0.1)

            high = min(row_bin.shape[0], 2**31 - 1)
            sample_ind = np.random.randint(
                low=0, high=high, size=self.batch_size, dtype=np.int32
            )

            row_ind = row_bin[sample_ind]
            col_ind = col_bin[sample_ind]

            dissimilarity_vals = self.dissimilarity_map[row_ind, col_ind]

            pixel_glimpses_moving, pixel_glimpses_fixed = self.get_glimpses(
                row_ind, col_ind
            )

            row_norm = row_ind / self.image_height
            col_norm = col_ind / self.image_width

            with tf.GradientTape() as tape:
                vec, sig = self.forward_pass(row_norm, col_norm, dissimilarity_vals)
                output_offset = self.vec_to_field(vec, sig)
                ssim_loss = self.compute_loss(
                    output_offset, pixel_glimpses_moving, pixel_glimpses_fixed
                )
                loss = ssim_loss
                if self.smoothness_weight > 0:
                    loss = loss + self.smoothness_weight * self._smoothness_loss(vec, row_norm, col_norm)
                if self.rigidity_weight > 0:
                    loss = loss + self.rigidity_weight * self._rigidity_loss(vec, row_norm, col_norm)

            gradients = tape.gradient(loss, self.trainable_variables)
            optimizer.apply_gradients(zip(gradients, self.trainable_variables))

            step_count += 1
            loss_accum += loss.numpy()
            ssim_val = ssim_loss.numpy()

            if verbose > 0 and step % verbose == 0:
                tq.set_description(f"{loss_accum / step_count:.5f}")
                tq.refresh()
                loss_accum = 0.0
                step_count = 0

            # --- Early stopping logic ---
            if early_stop_patience > 0:
                ssim_window.append(ssim_val)
                if len(ssim_window) > window_size:
                    ssim_window.pop(0)

                # Only evaluate after we have a full window
                if len(ssim_window) == window_size:
                    avg_ssim = sum(ssim_window) / window_size
                    if avg_ssim < best_ssim_loss:
                        best_ssim_loss = avg_ssim
                        best_weights = [w.numpy().copy() for w in self.trainable_variables]
                        steps_without_improvement = 0
                    else:
                        steps_without_improvement += 1

                    if steps_without_improvement >= early_stop_patience:
                        tq.set_description(f"Early stop at step {step} (best SSIM loss: {best_ssim_loss:.5f})")
                        tq.close()
                        break

        # Restore best weights
        if early_stop_patience > 0 and best_weights is not None:
            for var, w in zip(self.trainable_variables, best_weights):
                var.assign(w)

        return 0


    # -----------------------------------------------------------------------
    # Forward pass
    # -----------------------------------------------------------------------

    @tf.function
    def forward_pass(
        self,
        row_norm: tf.Tensor,
        col_norm: tf.Tensor,
        dissimilarity_input: tf.Tensor,
    ):
        """
        Run the MLP to predict displacement vectors and confidence scores.

        Parameters
        ----------
        row_norm : tf.Tensor, shape (B,)
            Row indices normalised to [0, 1].
        col_norm : tf.Tensor, shape (B,)
            Column indices normalised to [0, 1].
        dissimilarity_input : tf.Tensor, shape (B,)
            Local dissimilarity values at each glimpse centre.

        Returns
        -------
        vec : tf.Tensor, shape (B, 2)   — displacement in [-1, 1]
        sig : tf.Tensor, shape (B, 1)   — confidence scale
        """
        # Map [0, 1] → [-1, 1]
        coords_norm = tf.stack(
            [2.0 * row_norm - 1.0, 2.0 * col_norm - 1.0], axis=1
        )

        pos_encoded = get_positional_encoding(coords_norm, L=self.pos_encoding_L)

        output = tf.concat(
            [
                tf.cast(coords_norm, tf.float32),
                tf.cast(dissimilarity_input[:, None], tf.float32),
                tf.cast(pos_encoded, tf.float32),
            ],
            axis=1,
        )

        for layer in self.layers_custom:
            output = tf.nn.silu(layer(output))

        vec_output = tf.nn.tanh(self.vec_layer(output))
        sig_output = (
            tf.nn.softplus(self.sig_layer(output)) + 0.001
        ) / tf.math.exp(self.log_sig_scale_factor)

        return vec_output, sig_output

    # -----------------------------------------------------------------------
    # Displacement field
    # -----------------------------------------------------------------------

    @tf.function
    def vec_to_field(self, vec: tf.Tensor, sig: tf.Tensor) -> tf.Tensor:
        """
        Convert predicted vectors and confidence scores into a soft
        probability distribution over the (2*offset+1)^2 displacement grid.

        Parameters
        ----------
        vec : tf.Tensor, shape (B, 2)
        sig : tf.Tensor, shape (B, 1)

        Returns
        -------
        tf.Tensor, shape (B, 2*offset+1, 2*offset+1)
        """
        epsilon = 1e-6
        grid = tf.cast(
            tf.linspace(-1.0, 1.0, 2 * self.offset + 1), dtype=tf.float32
        )

        # Squared distance from predicted shift along each axis
        # Shapes: (B, 2*offset+1)
        row_dist = tf.square(grid[None, :] - vec[:, 0][:, None]) / (sig + epsilon)
        col_dist = tf.square(grid[None, :] - vec[:, 1][:, None]) / (sig + epsilon)

        # Outer sum → (B, 2*offset+1, 2*offset+1)
        field = row_dist[:, :, None] + col_dist[:, None, :]

        # --- LOGIC CHANGE (bug fix) ---
        # Original code used `vec.shape[0]` (static Python int) inside
        # @tf.function, which fails when the batch dimension is dynamic (None
        # at graph-tracing time).  We use `tf.shape(vec)[0]` instead so that
        # the reshape always uses the runtime batch size.
        batch = tf.shape(vec)[0]
        n = 2 * self.offset + 1
        field = tf.reshape(
            tf.nn.softmax(-tf.reshape(field, [batch, -1]), axis=-1),
            [batch, n, n],
        )

        return field

    # -----------------------------------------------------------------------
    # Loss
    # -----------------------------------------------------------------------

    @tf.function
    def compute_loss(
        self,
        output_offset: tf.Tensor,
        pixel_glimpses_moving: tf.Tensor,
        pixel_glimpses_fixed: tf.Tensor,
    ) -> tf.Tensor:
        """Apply the displacement field and compute the training loss."""
        # Convolve the moving glimpse with the predicted displacement kernel
        pixel_glimpses_moving = tf.nn.depthwise_conv2d(
            pixel_glimpses_moving,
            tf.transpose(output_offset, [1, 2, 0])[:, :, :, None],
            strides=[1, 1, 1, 1],
            padding="VALID",
        )

        if self.loss_type == "SSIM":
            return self._ssim_loss(pixel_glimpses_moving, pixel_glimpses_fixed)

        align_pixels = tf.reshape(pixel_glimpses_moving, [self.num_images, -1])
        ref_pixels = tf.reshape(pixel_glimpses_fixed, [self.num_images, -1])

        if self.loss_type == "MSE":
            return self._mse_loss(ref_pixels, align_pixels)
        elif self.loss_type == "NormCorr":
            return self._norm_corr_loss(ref_pixels, align_pixels)
        else:
            return self._corr_loss(ref_pixels, align_pixels)
        
    @tf.function
    def _ssim_loss(self, align_glimpses: tf.Tensor, ref_glimpses: tf.Tensor) -> tf.Tensor:
        pad = self.pad
        align = tf.reshape(
            tf.transpose(align_glimpses, [0, 3, 1, 2]),
            [-1, pad * 2 + 1, pad * 2 + 1],
        )
        ref = tf.reshape(
            tf.transpose(ref_glimpses, [0, 3, 1, 2]),
            [-1, pad * 2 + 1, pad * 2 + 1],
        )

        if self.loss_type == "MultiSSIM" and pad >= 10:
            # MS-SSIM needs minimum ~11px at each scale
            # With pad=25 (51x51): supports 3 scales (51→25→12)
            # With pad=12 (25x25): supports 2 scales (25→12)
            max_scales = 1
            size = pad * 2 + 1
            while size // 2 >= 11:
                max_scales += 1
                size //= 2
            # power_factors weights: last entry = luminance, all others = contrast+structure
            # Default (0.0448, 0.2856, 0.3001, 0.2363, 0.1333) for 5 scales
            # Truncate to however many scales we can support
            default_weights = [0.0448, 0.2856, 0.3001, 0.2363, 0.1333]
            weights = default_weights[:max_scales]
            # Renormalise so they sum to 1
            w_sum = sum(weights)
            weights = [w / w_sum for w in weights]

            val = tf.reshape(
                tf.image.ssim_multiscale(
                    ref[:, :, :, None], align[:, :, :, None],
                    max_val=1.0, power_factors=weights,
                ),
                [self.num_images, -1],
            )
        else:
            val = tf.reshape(
                tf.image.ssim(
                    ref[:, :, :, None], align[:, :, :, None], max_val=1.0,
                ),
                [self.num_images, -1],
            )

        return -tf.reduce_mean(val * self.coeff[:, None])


    @tf.function
    def _smoothness_loss(self, vec: tf.Tensor, row_norm: tf.Tensor, col_norm: tf.Tensor) -> tf.Tensor:
        """
        Penalise displacement differences between nearby pixel pairs.

        For each pixel in the batch, we also evaluate the network at a
        random neighbour within smoothness_radius.  The loss is the mean
        squared difference between their predicted displacement vectors.
        This enforces the prior that nearby pixels share similar shifts.
        """
        row_norm = tf.cast(row_norm, tf.float32)
        col_norm = tf.cast(col_norm, tf.float32)

        radius_row = tf.cast(self.smoothness_radius, tf.float32) / tf.cast(self.image_height, tf.float32)
        radius_col = tf.cast(self.smoothness_radius, tf.float32) / tf.cast(self.image_width, tf.float32)

        batch = tf.shape(row_norm)[0]
        dr = tf.random.uniform([batch], -radius_row, radius_row)
        dc = tf.random.uniform([batch], -radius_col, radius_col)

        # Clamp neighbour coords to [0, 1]
        row_nb = tf.clip_by_value(row_norm + dr, 0.0, 1.0)
        col_nb = tf.clip_by_value(col_norm + dc, 0.0, 1.0)

        # Look up dissimilarity at neighbour coords (nearest pixel)
        row_px = tf.cast(tf.round(row_nb * tf.cast(self.image_height - 1, tf.float32)), tf.int32)
        col_px = tf.cast(tf.round(col_nb * tf.cast(self.image_width - 1, tf.float32)), tf.int32)
        dissim_nb = tf.gather_nd(self.dissimilarity_map, tf.stack([row_px, col_px], axis=1))

        vec_nb, _ = self.forward_pass(row_nb, col_nb, dissim_nb)

        return tf.reduce_mean(tf.square(vec - vec_nb))
    
    @tf.function
    def _rigidity_loss(self, vec: tf.Tensor, row_norm: tf.Tensor, col_norm: tf.Tensor) -> tf.Tensor:
        """
        Penalise local area change by enforcing det(Jacobian) ≈ 1.

        Uses finite differences: evaluates the network at (r+eps, c) and
        (r, c+eps) to estimate the four partial derivatives of the
        displacement field, then computes det(J) and penalises deviation
        from 1.
        """
        row_norm = tf.cast(row_norm, tf.float32)
        col_norm = tf.cast(col_norm, tf.float32)

        # One-pixel step in normalised coords
        eps_r = 1.0 / tf.cast(self.image_height, tf.float32)
        eps_c = 1.0 / tf.cast(self.image_width, tf.float32)

        row_dr = tf.clip_by_value(row_norm + eps_r, 0.0, 1.0)
        col_dc = tf.clip_by_value(col_norm + eps_c, 0.0, 1.0)

        # Dissimilarity lookup helper
        def _dissim_at(r, c):
            rp = tf.cast(tf.round(r * tf.cast(self.image_height - 1, tf.float32)), tf.int32)
            cp = tf.cast(tf.round(c * tf.cast(self.image_width - 1, tf.float32)), tf.int32)
            return tf.gather_nd(self.dissimilarity_map, tf.stack([rp, cp], axis=1))

        vec_dr, _ = self.forward_pass(row_dr, col_norm, _dissim_at(row_dr, col_norm))
        vec_dc, _ = self.forward_pass(row_norm, col_dc, _dissim_at(row_norm, col_dc))

        # Partial derivatives of displacement
        du_dr = (vec_dr[:, 0] - vec[:, 0]) / eps_r
        dv_dr = (vec_dr[:, 1] - vec[:, 1]) / eps_r
        du_dc = (vec_dc[:, 0] - vec[:, 0]) / eps_c
        dv_dc = (vec_dc[:, 1] - vec[:, 1]) / eps_c

        # det(J) = (1 + du/dr)(1 + dv/dc) - (du/dc)(dv/dr)
        det_J = (1.0 + du_dr) * (1.0 + dv_dc) - du_dc * dv_dr

        return tf.reduce_mean(tf.square(det_J - 1.0))



    # -----------------------------------------------------------------------
    # Magnitude heatmap (diagnostic)
    # -----------------------------------------------------------------------

    def create_magnitude_heatmap(self) -> np.ndarray:
        """
        Compute a dense displacement-magnitude heatmap at the training resolution.

        Returns
        -------
        np.ndarray, shape (H, W)
            Per-pixel displacement magnitude in pixels.
        """
        h, w = self.image_height, self.image_width
        # indexing="ij" → first output varies along rows, second along cols
        row_coords, col_coords = np.meshgrid(
            np.arange(h), np.arange(w), indexing="ij"
        )

        row_norm = row_coords.flatten().astype("float32") / h
        col_norm = col_coords.flatten().astype("float32") / w
        dissim_flat = self.dissimilarity_map[
            row_coords.flatten(), col_coords.flatten()
        ]

        batch_size = 4096
        all_vecs = []
        for i in trange(0, len(row_norm), batch_size, leave=False, desc="Generating heatmap"):
            vec_batch, _ = self.forward_pass(
                tf.constant(row_norm[i : i + batch_size]),
                tf.constant(col_norm[i : i + batch_size]),
                tf.constant(dissim_flat[i : i + batch_size]),
            )
            all_vecs.append(vec_batch.numpy())

        vec_field = np.concatenate(all_vecs, axis=0)
        magnitude = np.linalg.norm(vec_field, axis=-1) * self.offset
        return magnitude.reshape(h, w)

    # -----------------------------------------------------------------------
    # Transform computation and application
    # -----------------------------------------------------------------------

    def compute_transform(self, num_cut: int = 100_000) -> int:
        """
        Compute the dense pixel-level warp field at full (pre-pool) resolution.

        The result is stored in ``self.full_transform`` as an (H, W, 2) array
        of source coordinates (suitable for ``skimage.transform.warp``).

        Parameters
        ----------
        num_cut : int
            Number of pixels processed per batch to limit GPU memory use.

        Returns
        -------
        int
            0 on success.
        """
        pool = self.pool
        h_pooled, w_pooled = self.image_height, self.image_width
        h_full, w_full = h_pooled * pool, w_pooled * pool

        # Full-resolution pixel coordinates
        row_v = np.arange(h_full)
        col_v = np.arange(w_full)
        pixel_ind = np.stack(
            np.meshgrid(row_v, col_v, indexing="ij"), axis=-1
        ).reshape(-1, 2)   # (H*W, 2) — columns: [row, col]

        row_norm_all = pixel_ind[:, 0] / float(h_full)
        col_norm_all = pixel_ind[:, 1] / float(w_full)

        # Dissimilarity map is at pooled resolution
        dissim_all = self.dissimilarity_map[
            pixel_ind[:, 0] // pool,
            pixel_ind[:, 1] // pool,
        ]

        pixel_transform = np.zeros((pixel_ind.shape[0], 2), dtype=np.float32)

        t0 = time.time()
        for start in trange(
            0, pixel_ind.shape[0], num_cut, leave=True, desc="Computing transform"
        ):
            end = min(start + num_cut, pixel_ind.shape[0])
            vec_batch, _ = self.forward_pass(
                tf.constant(row_norm_all[start:end], dtype=tf.float32),
                tf.constant(col_norm_all[start:end], dtype=tf.float32),
                tf.constant(dissim_all[start:end], dtype=tf.float32),
            )
            # vec is in [-1, 1]; scale to full-resolution pixels
            pixel_transform[start:end] = vec_batch.numpy() * self.offset * pool

        print(f"Transform loop: {time.time() - t0:.4f}s")

        # vec[:, 0] = row displacement, vec[:, 1] = col displacement
        # Source coords = destination coords - displacement
        new_ind = pixel_ind.astype(np.float32) - pixel_transform
        new_ind = new_ind.reshape(h_full, w_full, 2)
        new_ind[:, :, 0] = np.clip(new_ind[:, :, 0], 0, h_full - 1)
        new_ind[:, :, 1] = np.clip(new_ind[:, :, 1], 0, w_full - 1)

        self.full_transform = new_ind

        if self.save_transform_file_path:
            np.save(self.save_transform_file_path, self.full_transform)

        return 0

    def apply_transform(self, image: np.ndarray) -> np.ndarray:
        """
        Apply the computed warp field to an image.

        Parameters
        ----------
        image : np.ndarray
            Image to warp. Must match the full (pre-pool) resolution.

        Returns
        -------
        np.ndarray
            Warped image (float32, same shape as input).
        """
        return skimage.transform.warp(
            image.astype("float32"),
            np.array(
                [self.full_transform[:, :, 0], self.full_transform[:, :, 1]]
            ),
            order=0,
        )

    def get_transform(self) -> np.ndarray:
        """Return the stored warp field (H, W, 2)."""
        return self.full_transform
