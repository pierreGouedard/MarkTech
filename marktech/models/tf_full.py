# Global import
from typing import Tuple, Optional, Dict, Any, Union
from pathlib import Path
import json
from matplotlib import pyplot as plt
import numpy as np
import tensorflow as tf

# Local import
from .datasets import DatasetMeta


class TfModelFull:
    """Predict PHQ-8 scores indicating depression from 1D feature."""

    def __init__(
            self, meta_dataset: DatasetMeta, learning_rate: float, nb_epoch: int, batch_size: int = 32,
            threshold: float = 0.5, dropout: float = None, sigmoid_last: bool = True,
            network: Optional[tf.keras.Model] = None
    ):
        # training params
        self.learning_rate, self.nb_epoch, self.batch_size = learning_rate, nb_epoch, batch_size
        self.dropout, self.sigmoid_last = dropout, sigmoid_last
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)

        # Meta inputs
        self.meta_dataset = meta_dataset

        # Create and compile model
        if network is None:
            self.network = self.__create_network(
                self.meta_dataset.input_dim, self.meta_dataset.norm_data, self.dropout,
                self.sigmoid_last
            )
            self.network.compile(optimizer=self.optimizer, loss=tf.keras.losses.MeanSquaredError())
        else:
            self.network = network

        # Evaluation attribute
        self.history = None
        self.threshold = threshold

    @staticmethod
    def from_dict(d_params: Dict[str, Any], meta_dataset: DatasetMeta, network: Optional[tf.keras.Model] = None):
        return TfModelFull(meta_dataset, network=network, **d_params)

    @staticmethod
    def load(path_dir: Path) -> 'TfModelFull':
        with open(path_dir / 'params.json', 'r') as handle:
            d_params = json.load(handle)

        meta_ds = DatasetMeta().load(path_dir / 'meta')

        # Load Keras model
        network = tf.keras.models.load_model(path_dir / 'network')

        return TfModelFull(**{**d_params, 'network': network, 'meta_dataset': meta_ds})

    def to_dict(self):
        return {
            'learning_rate': self.learning_rate, 'batch_size': self.batch_size, 'nb_epoch': self.nb_epoch,
            'threshold': self.threshold, 'dropout': self.dropout, 'sigmoid_last': self.sigmoid_last
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
            input_dim: Tuple[int, int], norm_data: np.ndarray, dropout: float, sigmoid_last: bool
    ) -> tf.keras.Model:
        """Create forward network.

        Architecture:
        (DENSE -> RELU){x2} -> DENSE -> [SIGMOID] -> OUTPUT

        inputs dim are (n_batch, n_lld) [param input doesn't include n_batch].

        """
        X_input = tf.keras.layers.Input(input_dim)

        # Start with normalization
        if norm_data is not None:
            layer = tf.keras.layers.Normalization(axis=None)
            layer.adapt(norm_data)
            X = layer(X_input)
        else:
            X = X_input

        # 1st dense FC layer
        X = tf.keras.layers.Dense(int(input_dim[0] / 2), name='hidden_layer_1', activation="relu")(X)

        if dropout is not None:
            X = tf.keras.layers.Dropout(dropout)(X)

        # 2nd dense FC layer
        X = tf.keras.layers.Dense(int(input_dim[0] / 4), name='hidden_layer_2', activation="relu")(X)

        # End with a sigmoid output layer
        if sigmoid_last:
            X = tf.keras.layers.Dense(1, name='output_layer', activation='sigmoid')(X)

        network = tf.keras.Model(inputs=X_input, outputs=X, name='model_avg_lld')

        return network

    def fit(
            self, train_datasets: tf.data.Dataset, show_eval: bool = False
    ) -> 'TfModelFull':
        """
        Fit network using early stopping on validation dataset.

        Args:
            train_datasets: tf.python.data.Dataset
            show_eval: bool

        Returns:
        None
        """
        # Build datasets
        train_dataset, val_dataset = self.build_datasets(train_datasets)

        # Create early stopping callback
        es = tf.keras.callbacks.EarlyStopping(monitor='val_loss', mode='min', verbose=0, patience=4)
        lr = tf.keras.callbacks.ReduceLROnPlateau(factor=0.1, patience=1, verbose=0)

        # fit the model
        self.history = self.network.fit(
            train_dataset, validation_data=val_dataset, epochs=self.nb_epoch, callbacks=[lr, es], verbose=0
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
        # Fit and evaluate model

        """
        tf_y = self.network(X)
        return np.array(tf_y.numpy()[:, 0])

    def build_datasets(
            self, fulltrain_dataset: tf.data.Dataset
    ) -> Tuple[tf.data.Dataset, tf.data.Dataset]:
        """Define batch size and prefect size of datasets for fitting.

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
