from __future__ import annotations

import tensorflow as tf
from tensorflow.keras import layers, models


# ============================================================
# Graph Convolution Layer
# ============================================================

@tf.keras.utils.register_keras_serializable(package="BrainMRI")
class GraphConvolution(layers.Layer):

    def __init__(self, units, adjacency, **kwargs):
        super().__init__(**kwargs)
        self.units = units
        self.adjacency = tf.convert_to_tensor(
            adjacency,
            dtype=tf.float32
        )

    def build(self, input_shape):

        self.kernel = self.add_weight(
            name="kernel",
            shape=(int(input_shape[-1]), self.units),
            initializer="glorot_uniform",
            trainable=True
        )

        self.bias = self.add_weight(
            name="bias",
            shape=(self.units,),
            initializer="zeros",
            trainable=True
        )

        super().build(input_shape)

    def call(self, inputs):

        aggregated = tf.einsum(
            "ij,bjf->bif",
            self.adjacency,
            inputs
        )

        return tf.nn.gelu(
            tf.matmul(aggregated, self.kernel)
            + self.bias
        )

    def get_config(self):

        config = super().get_config()

        config.update({
            "units": self.units,
            "adjacency": self.adjacency.numpy().tolist()
        })

        return config


# ============================================================
# Position Embedding
# ============================================================

@tf.keras.utils.register_keras_serializable(package="BrainMRI")
class TokenPositionEmbedding(layers.Layer):

    def __init__(
        self,
        token_count,
        embedding_dim,
        **kwargs
    ):

        super().__init__(**kwargs)

        self.token_count = token_count
        self.embedding_dim = embedding_dim

        self.position_embedding = layers.Embedding(
            input_dim=token_count,
            output_dim=embedding_dim
        )

    def call(self, inputs):

        positions = tf.range(
            start=0,
            limit=self.token_count
        )

        return inputs + self.position_embedding(positions)

    def get_config(self):

        config = super().get_config()

        config.update({
            "token_count": self.token_count,
            "embedding_dim": self.embedding_dim
        })

        return config


# ============================================================
# Create Graph Adjacency Matrix
# ============================================================

def make_grid_adjacency(grid_size):

    node_count = grid_size * grid_size

    adjacency = tf.eye(
        node_count,
        dtype=tf.float32
    )

    for row in range(grid_size):

        for column in range(grid_size):

            node = row * grid_size + column

            for row_delta in (-1, 0, 1):

                for column_delta in (-1, 0, 1):

                    if row_delta == 0 and column_delta == 0:
                        continue

                    neighbor_row = row + row_delta
                    neighbor_column = column + column_delta

                    if (
                        0 <= neighbor_row < grid_size
                        and
                        0 <= neighbor_column < grid_size
                    ):

                        neighbor = (
                            neighbor_row * grid_size
                            + neighbor_column
                        )

                        adjacency = tf.tensor_scatter_nd_update(
                            adjacency,
                            [[node, neighbor]],
                            [1.0]
                        )

    degree = tf.reduce_sum(
        adjacency,
        axis=1
    )

    normalization = tf.math.rsqrt(degree)

    return (
        normalization[:, None]
        * adjacency
        * normalization[None, :]
    )


# ============================================================
# GCN + Transformer Model
# ============================================================

def build_gcn_transformer(
    input_shape=(128, 128, 3),
    patch_size=8,
    embedding_dim=96,
    num_heads=4,
    dropout=0.25,
    num_classes=4
):

    height, width, _ = input_shape

    if height % patch_size != 0:
        raise ValueError(
            "Image height must be divisible by patch_size"
        )

    if width % patch_size != 0:
        raise ValueError(
            "Image width must be divisible by patch_size"
        )

    if patch_size % 4 != 0:
        raise ValueError(
            "patch_size must be divisible by 4 for the convolutional feature stem"
        )

    grid_height = height // patch_size
    grid_width = width // patch_size

    if grid_height != grid_width:
        raise ValueError(
            "Image must create a square patch grid"
        )

    token_count = grid_height * grid_width

    adjacency = make_grid_adjacency(
        grid_height
    )

    # --------------------------------------------------------
    # INPUT
    # --------------------------------------------------------

    inputs = layers.Input(
        shape=input_shape,
        name="mri_image"
    )

    # --------------------------------------------------------
    # NORMALIZATION
    # --------------------------------------------------------

    x = layers.Rescaling(
        1.0 / 255.0,
        name="rescale"
    )(inputs)

    # Learn local MRI features before forming graph nodes and transformer tokens.
    x = layers.Conv2D(
        32,
        kernel_size=3,
        strides=2,
        padding="same",
        use_bias=False,
        name="local_feature_stem_1",
    )(x)

    x = layers.BatchNormalization(
        name="local_feature_norm_1"
    )(x)

    x = layers.Activation(
        "gelu",
        name="local_feature_activation_1",
    )(x)

    x = layers.Conv2D(
        embedding_dim,
        kernel_size=3,
        strides=2,
        padding="same",
        use_bias=False,
        name="local_feature_stem_2",
    )(x)

    x = layers.BatchNormalization(
        name="local_feature_norm_2"
    )(x)

    x = layers.Activation(
        "gelu",
        name="local_feature_activation_2",
    )(x)

    # --------------------------------------------------------
    # PATCH PROJECTION
    # --------------------------------------------------------

    stem_patch_size = patch_size // 4
    x = layers.Conv2D(
        embedding_dim,
        kernel_size=stem_patch_size,
        strides=stem_patch_size,
        padding="valid",
        name="patch_projection"
    )(x)

    # --------------------------------------------------------
    # PATCH TOKENS
    # --------------------------------------------------------

    x = layers.Reshape(
        (
            token_count,
            embedding_dim
        ),
        name="patch_tokens"
    )(x)

    # ========================================================
    # GCN BLOCK
    # ========================================================

    x = GraphConvolution(
        embedding_dim,
        adjacency,
        name="gcn_1"
    )(x)

    x = layers.LayerNormalization(
        name="gcn_norm_1"
    )(x)

    x = layers.Dropout(
        dropout
    )(x)

    x = GraphConvolution(
        embedding_dim,
        adjacency,
        name="gcn_2"
    )(x)

    x = layers.LayerNormalization(
        name="gcn_norm_2"
    )(x)

    # ========================================================
    # POSITION EMBEDDING
    # ========================================================

    x = TokenPositionEmbedding(
        token_count,
        embedding_dim,
        name="position_embedding"
    )(x)

    # ========================================================
    # TRANSFORMER
    # ========================================================

    attention = layers.MultiHeadAttention(
        num_heads=num_heads,
        key_dim=embedding_dim // num_heads,
        dropout=dropout,
        name="transformer_attention"
    )(x, x)

    x = layers.LayerNormalization(
        name="transformer_norm_1"
    )(x + attention)

    # --------------------------------------------------------
    # FEED FORWARD NETWORK
    # --------------------------------------------------------

    ff = layers.Dense(
        embedding_dim * 2,
        activation="gelu"
    )(x)

    ff = layers.Dropout(
        dropout
    )(ff)

    ff = layers.Dense(
        embedding_dim
    )(ff)

    x = layers.LayerNormalization(
        name="transformer_norm_2"
    )(x + ff)

    # ========================================================
    # CLASSIFICATION
    # ========================================================

    x = layers.GlobalAveragePooling1D(
        name="global_pooling"
    )(x)

    x = layers.Dense(
        128,
        activation="gelu",
        name="feature_layer"
    )(x)

    x = layers.Dropout(
        dropout
    )(x)

    # ========================================================
    # 4 CLASS OUTPUT
    # ========================================================

    outputs = layers.Dense(
        num_classes,
        activation="softmax",
        name="tumor_class"
    )(x)

    # ========================================================
    # MODEL
    # ========================================================

    model = models.Model(
        inputs=inputs,
        outputs=outputs,
        name="gcn_transformer_brain_tumor"
    )

    return model