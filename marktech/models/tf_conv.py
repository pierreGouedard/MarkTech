# Global import
from typing import Tuple, Optional, Dict, Any, Union
from pathlib import Path
import json
from matplotlib import pyplot as plt
import numpy as np
import tensorflow as tf

# Local import
from .datasets import DatasetMeta


class CustomAttention(tf.keras.layers.Layer):
    def __init__(self, **kwargs):
        super(CustomAttention, self).__init__(**kwargs)
        self.W, self.b = None, None

    def build(self, input_shape):
        self.W = self.add_weight(
            name='attention_weight', shape=(input_shape[-1], 1), initializer='random_normal', trainable=True
        )
        self.b = self.add_weight(
            name='attention_bias', shape=(input_shape[1], 1), initializer='zeros', trainable=True
        )
        super(CustomAttention, self).build(input_shape)

    def call(self, inputs, *args, **kwargs):
        # Alignment scores. Pass them through tanh function
        e = tf.keras.activations.tanh(tf.keras.backend.dot(inputs, self.W) + self.b)

        # Remove dimension of size 1
        e = tf.keras.backend.squeeze(e, axis=-1)

        # Compute the weights
        alpha = tf.keras.activations.softmax(e)

        # Reshape to tensorFlow format
        alpha = tf.keras.backend.expand_dims(alpha, axis=-1)

        # Compute the context vector
        context = inputs * alpha
        context = tf.keras.backend.sum(context, axis=1)

        return context


class TfModelConv:

    def __init__(
            self, meta_dataset: DatasetMeta, n_kernels: int, kernel_size: int,
            learning_rate: float, nb_epoch: int, batch_size: int = 32, dropout: float = None,
            attention: bool = True, network: Optional[tf.keras.Model] = None
    ):
        # training params
        self.learning_rate, self.nb_epoch, self.batch_size = learning_rate, nb_epoch, batch_size
        self.dropout, self.attention = dropout, attention
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)

        # Meta inputs
        self.meta_dataset = meta_dataset

        # Kernel's dim
        self.n_kernels, self.kernel_size = n_kernels, kernel_size

        # Create and compile model
        if network is None:
            self.network = self.__create_network(
                self.n_kernels, self.kernel_size, self.meta_dataset.input_dim, self.meta_dataset.norm_data,
                self.dropout, self.attention
            )
            self.network.compile(optimizer=self.optimizer, loss=tf.keras.losses.MeanSquaredError())
        else:
            self.network = network

        # Evaluation attribute
        self.history = None

    @staticmethod
    def from_dict(d_params: Dict[str, Any], meta_dataset: DatasetMeta, network: Optional[tf.keras.Model] = None):
        return TfModelConv(meta_dataset, network=network, **d_params)

    @staticmethod
    def load(path_dir: Path):
        with open(path_dir / 'params.json', 'r') as handle:
            d_params = json.load(handle)

        meta_ds = DatasetMeta().load(path_dir / 'meta')

        # Load Keras model
        network = tf.keras.models.load_model(path_dir / 'network')

        return TfModelConv(**{**d_params, 'network': network, 'meta_dataset': meta_ds})

    def to_dict(self):
        return {
            'n_kernels': self.n_kernels, 'kernel_size': self.kernel_size, 'learning_rate': self.learning_rate,
            'batch_size': self.batch_size, 'nb_epoch': self.nb_epoch, 'dropout': self.dropout,
            'attention': self.attention
        }

    def save(self, path_dir: Path):
        path_dir.mkdir()
        d_params = self.to_dict()
        with open(path_dir / 'params.json', 'w') as handle:
            json.dump(d_params, handle)

        # save meta
        self.meta_dataset.save(path_dir / 'meta')

        # Save keras model
        self.network.save(path_dir / 'network')

    @staticmethod
    def __create_network(
            n_kernels: int, kernel_size: int, input_dim: Tuple[int, int], norm_data: np.ndarray, dropout: float,
            attention: bool
    ) -> tf.keras.Model:
        """Create forward network.

        Architecture:
        CONV1D -> RELU -> [Attention | (MAXPOOL -> DENSE -> RELU] -> (DENSE -> RELU) -> DENSE -> [SIGMOID] -> OUTPUT

        inputs dim are (n_batch, n_sequence, (n_lld, n_vgg)) [param 'input_dim' doesn't include n_batch].

        """
        # Start with normalization
        X_input = tf.keras.layers.Input(input_dim)

        # Start with normalization if needed
        if norm_data is not None:
            layer = tf.keras.layers.Normalization(axis=None)
            layer.adapt(norm_data)
            X = layer(X_input)
        else:
            X = X_input

        # 1D conv with activation
        X = tf.keras.layers.Conv1D(
            n_kernels, kernel_size, activation='relu', padding="same", input_shape=input_dim
        )(X)
        if dropout is not None:
            X = tf.keras.layers.Dropout(dropout)(X)

        if not attention:
            # Max pool layer + flatten + FC layer
            X = tf.keras.layers.MaxPooling1D(pool_size=input_dim[0], padding='valid')(X)
            X = tf.keras.layers.Flatten()(X)
            X = tf.keras.layers.Dense(int(n_kernels / 2), name='hidden_dense_layer_1', activation="relu")(X)
        else:
            X = CustomAttention()(X)

        # 2nd dense FC layer
        X = tf.keras.layers.Dense(int(n_kernels / 4), name='hidden_dense_layer_2', activation="relu")(X)

        # End with a sigmoid output layer
        network = tf.keras.Model(inputs=X_input, outputs=X, name='model_seq_lld')

        return network

    def fit(
            self, train_datasets: tf.data.Dataset, show_eval: bool = False
    ) -> 'TfModelConv':
        """
        Fit network using early stopping on validation dataset.

        Args:
            train_datasets: tf.python.data.Dataset
            show_eval: bool

        Returns:
        self
        """
        # Build datasets
        train_dataset, val_dataset = self.build_datasets(train_datasets)

        # Create callbacks
        es = tf.keras.callbacks.EarlyStopping(monitor='val_loss', mode='min', verbose=0, patience=4)
        lr = tf.keras.callbacks.ReduceLROnPlateau(factor=0.5, patience=1, verbose=0)

        # fit the model
        self.history = self.network.fit(
            train_dataset, validation_data=val_dataset, epochs=self.nb_epoch, callbacks=[lr, es], verbose=1
        )

        if show_eval:
            self.evaluate_training(train_dataset, val_dataset)

        return self

    def evaluate_training(self, train_dataset: tf.data.Dataset, val_dataset: tf.data.Dataset) -> None:
        """
        Display performance of model on train/val and history of metrics through epochs.

        Args:
            train_dataset: tf.data.Dataset
                Training dataset.
            val_dataset: tf.data.Dataset
                Validation dataset

        Returns:
        None
        """
        # Get metrics
        train_ce = self.network.evaluate(train_dataset, verbose=0)
        val_ce = self.network.evaluate(val_dataset, verbose=0)

        print('Cross entropy error: Train: %.3f, Test: %.3f' % (val_ce, train_ce))

        # plot training history
        plt.plot(self.history.history['loss'], label='train')
        plt.plot(self.history.history['val_loss'], label='validation')
        plt.legend()
        plt.show()

    def predict(self, X: Union[tf.Tensor, np.ndarray]) -> np.ndarray:
        """ Predict new inputs with network

        """
        tf_y = self.network(X)
        return np.array(tf_y.numpy()[:, 0])

    def build_datasets(
            self, fulltrain_dataset: tf.data.Dataset
    ) -> Tuple[tf.data.Dataset, tf.data.Dataset]:
        """Define batch size and prefecth size of datasets for fitting.

        """
        train_dataset = (
            fulltrain_dataset
            .skip(self.meta_dataset.val_before)
            .batch(self.batch_size, drop_remainder=False)
            .prefetch(int(self.batch_size / 2))
        )

        val_dataset = (
            fulltrain_dataset
            .take(self.meta_dataset.val_before)
            .batch(self.batch_size, drop_remainder=False)
            .prefetch(int(self.batch_size / 2))
        )

        return train_dataset, val_dataset



