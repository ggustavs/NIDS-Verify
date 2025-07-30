"""
Model architectures for NIDS
"""
import tensorflow as tf
from tensorflow import keras
from keras import layers
from typing import Dict, Any
from src.config import config
from src.utils.logging import get_logger
from src.utils.performance import get_model_info

logger = get_logger(__name__)


class ModelFactory:
    """Factory for creating different model architectures"""

    def __init__(self, input_size: int, seed: int = None):
        self.input_size = input_size
        self.seed = seed or config.model.initializer_seed
        self.initializer = tf.keras.initializers.GlorotUniform(seed=self.seed)

    def create_model(self, model_type: str) -> keras.Model:
        """Create a model based on the specified type"""
        model_creators = {
            'small': self._create_small_model,
            'mid': self._create_mid_model,
            'mid2': self._create_mid2_model,
            'mid3': self._create_mid3_model,
            'mid4': self._create_mid4_model,
            'big': self._create_big_model,
            'big2': self._create_big2_model,
            'big3': self._create_big3_model,
            'big4': self._create_big4_model,
            'massive': self._create_massive_model,
        }

        if model_type not in model_creators:
            raise ValueError(f"Unknown model type: {model_type}. Available: {list(model_creators.keys())}")

        logger.info(f"Creating {model_type} model with input size {self.input_size}")
        model = model_creators[model_type]()

        # Log model information
        model_info = get_model_info(model)
        logger.info(f"Model created - Parameters: {model_info.get('total_parameters', 'unknown'):,}")
        logger.debug(f"Model info: {model_info}")

        return model

    def _create_small_model(self) -> keras.Model:
        return keras.Sequential([
            layers.Input(shape=(self.input_size,), name='input_features'),
            layers.Dense(128, activation='relu', kernel_initializer=self.initializer, name='dense_1'),
            layers.Dense(2, activation='linear', kernel_initializer=self.initializer, name='output_layer')
        ], name='small_model')

    def _create_mid_model(self) -> keras.Model:
        return keras.Sequential([
            layers.Input(shape=(self.input_size,), name='input_features'),
            layers.Dense(256, activation='relu', kernel_initializer=self.initializer, name='dense_1'),
            layers.Dense(128, activation='relu', kernel_initializer=self.initializer, name='dense_2'),
            layers.Dense(2, activation='linear', kernel_initializer=self.initializer, name='output_layer')
        ], name='mid_model')

    def _create_mid2_model(self) -> keras.Model:
        return keras.Sequential([
            layers.Input(shape=(self.input_size,), name='input_features'),
            layers.Dense(512, activation='relu', kernel_initializer=self.initializer, name='dense_1'),
            layers.Dense(256, activation='relu', kernel_initializer=self.initializer, name='dense_2'),
            layers.Dense(128, activation='relu', kernel_initializer=self.initializer, name='dense_3'),
            layers.Dense(2, activation='linear', kernel_initializer=self.initializer, name='output_layer')
        ], name='mid2_model')

    def _create_mid3_model(self) -> keras.Model:
        return keras.Sequential([
            layers.Input(shape=(self.input_size,), name='input_features'),
            layers.Dense(512, activation='relu', kernel_initializer=self.initializer, name='dense_1'),
            layers.Dense(256, activation='relu', kernel_initializer=self.initializer, name='dense_2'),
            layers.Dense(256, activation='relu', kernel_initializer=self.initializer, name='dense_3'),
            layers.Dense(128, activation='relu', kernel_initializer=self.initializer, name='dense_4'),
            layers.Dense(2, activation='linear', kernel_initializer=self.initializer, name='output_layer')
        ], name='mid3_model')

    def _create_mid4_model(self) -> keras.Model:
        return keras.Sequential([
            layers.Input(shape=(self.input_size,), name='input_features'),
            layers.Dense(512, activation='relu', kernel_initializer=self.initializer, name='dense_0'),
            layers.Dense(512, activation='relu', kernel_initializer=self.initializer, name='dense_1'),
            layers.Dense(256, activation='relu', kernel_initializer=self.initializer, name='dense_2'),
            layers.Dense(128, activation='relu', kernel_initializer=self.initializer, name='dense_3'),
            layers.Dense(2, activation='linear', kernel_initializer=self.initializer, name='output_layer')
        ], name='mid4_model')

    def _create_big_model(self) -> keras.Model:
        return keras.Sequential([
            layers.Input(shape=(self.input_size,), name='input_features'),
            layers.Dense(1024, activation='relu', kernel_initializer=self.initializer, name='dense_1'),
            layers.Dense(512, activation='relu', kernel_initializer=self.initializer, name='dense_2'),
            layers.Dense(256, activation='relu', kernel_initializer=self.initializer, name='dense_3'),
            layers.Dense(128, activation='relu', kernel_initializer=self.initializer, name='dense_4'),
            layers.Dense(2, activation='linear', kernel_initializer=self.initializer, name='output_layer')
        ], name='big_model')

    def _create_big2_model(self) -> keras.Model:
        return keras.Sequential([
            layers.Input(shape=(self.input_size,), name='input_features'),
            layers.Dense(1024, activation='relu', kernel_initializer=self.initializer, name='dense_1'),
            layers.Dense(512, activation='relu', kernel_initializer=self.initializer, name='dense_2'),
            layers.Dense(512, activation='relu', kernel_initializer=self.initializer, name='dense_3'),
            layers.Dense(256, activation='relu', kernel_initializer=self.initializer, name='dense_4'),
            layers.Dense(2, activation='linear', kernel_initializer=self.initializer, name='output_layer')
        ], name='big2_model')

    def _create_big3_model(self) -> keras.Model:
        return keras.Sequential([
            layers.Input(shape=(self.input_size,), name='input_features'),
            layers.Dense(1024, activation='relu', kernel_initializer=self.initializer, name='dense_1'),
            layers.Dense(1024, activation='relu', kernel_initializer=self.initializer, name='dense_2'),
            layers.Dense(512, activation='relu', kernel_initializer=self.initializer, name='dense_3'),
            layers.Dense(256, activation='relu', kernel_initializer=self.initializer, name='dense_4'),
            layers.Dense(2, activation='linear', kernel_initializer=self.initializer, name='output_layer')
        ], name='big3_model')

    def _create_big4_model(self) -> keras.Model:
        return keras.Sequential([
            layers.Input(shape=(self.input_size,), name='input_features'),
            layers.Dense(2048, activation='relu', kernel_initializer=self.initializer, name='dense_1'),
            layers.Dense(1024, activation='relu', kernel_initializer=self.initializer, name='dense_2'),
            layers.Dense(512, activation='relu', kernel_initializer=self.initializer, name='dense_3'),
            layers.Dense(256, activation='relu', kernel_initializer=self.initializer, name='dense_4'),
            layers.Dense(2, activation='linear', kernel_initializer=self.initializer, name='output_layer')
        ], name='big4_model')

    def _create_massive_model(self) -> keras.Model:
        return keras.Sequential([
            layers.Input(shape=(self.input_size,), name='input_features'),
            layers.Dense(2048, activation='relu', kernel_initializer=self.initializer, name='dense_3'),
            layers.Dense(1024, activation='relu', kernel_initializer=self.initializer, name='dense_4'),
            layers.Dense(512, activation='relu', kernel_initializer=self.initializer, name='dense_5'),
            layers.Dense(256, activation='relu', kernel_initializer=self.initializer, name='dense_6'),
            layers.Dense(128, activation='relu', kernel_initializer=self.initializer, name='dense_7'),
            layers.Dense(2, activation='linear', kernel_initializer=self.initializer, name='output_layer')
        ], name='massive_model')


def create_model(input_size: int, model_type: str) -> keras.Model:
    """
    Create a model instance

    Args:
        input_size: Size of input features
        model_type: Type of model to create

    Returns:
        Keras model instance
    """
    factory = ModelFactory(input_size)
    model = factory.create_model(model_type)

    # Print model summary for debugging
    if logger.logger.level <= 10:  # DEBUG level
        model.summary()

    return model
