from typing import Union, Tuple, Optional, List, Callable
from metrics import (dice_coefficient_loss, get_label_dice_coefficient_function, dice_coefficient,
                     weighted_dice_coefficient, weighted_dice_coefficient_loss, get_metric_as_costome_object)
import tensorflow as tf


class Model3DUnet:
    """ модель 3D-UNet """

    # INPUT_IMG_SHAPE: Форма входных данных tuple(z_size, x_size, y_size) или int(size) если форма соответствует кубу.
    # Размеры x, y и z должны делиться на размер пула в степени глубины UNet, то есть pool_size ^ depth.
    INPUT_IMG_SHAPE: Union[tuple, int] = (128, 128, 128)
    # N_CHANNELS: количество каналов в входном изображении
    N_CHANNELS: int = 3
    # N_CLASSES: Количество классов, которые изучает модель. (количество меток + 1)
    N_CLASSES = 5
    # POOL_SIZE: Размер пула для максимальных операций объединения.
    POOL_SIZE = (2, 2, 2,)
    # INITIAL_LEARNING_RATE: Начальная скорость обучения модели.
    INITIAL_LEARNING_RATE = 0.00001
    # TYPE_UP_CONVOLUTION: тип повышающего слоя. up_sampling_3d использует меньше памяти.
    TYPE_UP_CONVOLUTION = ["conv_3d_transpose", "up_sampling_3d"]
    # DEPTH: глубина модели.
    DEPTH = 4
    # START_VAL_FILTERS: Количество фильтров, которые будет иметь первый слой в сверточной сети.
    # Следующие слои будут содержать число, кратное этому числу.
    START_VAL_FILTERS = 16
    # LIST_METRICS: Список метрик, которые будут вычисляться во время обучения модели (по умолчанию - Мера Сёренсена).
    LIST_METRICS = [dice_coefficient]

    def __init__(self,
                 input_img_shape: Union[Tuple[int, int, int], int, None] = None,
                 n_channels: Optional[int] = None,
                 n_classes: Optional[int] = None,
                 pool_size: Union[Tuple[int, int, int], int, None] = None,
                 initial_learning_rate: Optional[float] = None,
                 type_up_convolution: Optional[str] = None,
                 depth: Optional[int] = None,
                 start_val_filters: Optional[int] = None,
                 list_metrics: Optional[List[Callable]] = None):
        """

        Args:
            input_img_shape: Форма входных данных кортеж(3N) или целое число (если форма соответствует кубу).
            n_channels: Размер канала входных данных.
            n_classes: Количество классов, которые изучает модель (количество меток + 1).
            pool_size: Размер пула для максимальных операций объединения. Целое число или кортеж.
            initial_learning_rate: Начальная скорость обучения модели.
            type_up_convolution: тип повышающего слоя. up_sampling_3d использует меньше памяти.
            depth: глубина модели. Уменьшение глубины может уменьшить объем памяти, необходимый для тренировки.
            start_val_filters: Количество фильтров, которые будет иметь первый слой в сети.
            list_metrics: Список метрик, для обучения модели (по умолчанию - Мера Сёренсена).

        """
        if input_img_shape is None:
            self.input_img_shape = self.INPUT_IMG_SHAPE
        elif isinstance(input_img_shape, tuple):
            if len(input_img_shape) == 3:
                self.input_img_shape = input_img_shape
            else:
                raise TypeError(f"Атрибут input_img_shape ожидает тип tuple размером 3, "
                                f"но получил tuple размером {len(input_img_shape)}")
        elif isinstance(input_img_shape, int):
            self.input_img_shape = (input_img_shape, input_img_shape, input_img_shape,)
        else:
            raise TypeError(f"Атрибут input_img_shape ожидает тип Union[Tuple[int,int,int],int,None], "
                            f"но получил тип {type(input_img_shape)}")

        if n_channels is None:
            self.n_channels = self.N_CHANNELS
        elif isinstance(n_channels, int):
            self.n_channels = n_channels
        else:
            raise TypeError(f"Атрибут n_channels ожидает тип int, но получил тип {type(n_channels)}")

        if n_classes is None:
            self.n_classes = self.N_CLASSES
        elif isinstance(n_classes, int):
            self.n_classes = n_classes
        else:
            raise TypeError(f"Атрибут n_classes ожидает тип int, но получил тип {type(n_classes)}")

        if pool_size is None:
            self.pool_size = self.POOL_SIZE
        elif isinstance(pool_size, tuple):
            if len(pool_size) == 3:
                self.pool_size = pool_size
            else:
                raise TypeError(f"Атрибут pool_size ожидает тип tuple размером 3, "
                                f"но получил tuple размером {len(pool_size)}")
        elif isinstance(pool_size, int):
            self.pool_size = (pool_size, pool_size, pool_size,)
        else:
            raise TypeError(f"Атрибут pool_size ожидает тип Union[Tuple[int,int,int],int,None], "
                            f"но получил тип {type(pool_size)}")

        if initial_learning_rate is None:
            self.initial_learning_rate = self.INITIAL_LEARNING_RATE
        elif isinstance(initial_learning_rate, float):
            self.initial_learning_rate = initial_learning_rate
        else:
            raise TypeError(
                f"Атрибут initial_learning_rate ожидает тип float, но получил тип {type(initial_learning_rate)}")

        if type_up_convolution is None:
            self.type_up_convolution = self.TYPE_UP_CONVOLUTION[0]
        elif isinstance(type_up_convolution, str):
            if type_up_convolution in self.TYPE_UP_CONVOLUTION:
                self.type_up_convolution = type_up_convolution
            else:
                raise TypeError(f"Атрибут type_up_convolution ожидает, "
                                f"что значение будет соответствует одному из списка {self.TYPE_UP_CONVOLUTION}, "
                                f"но получил значение {type_up_convolution}")
        else:
            raise TypeError(f"Атрибут type_up_convolution ожидает тип str, "
                            f"но получил тип {type(type_up_convolution)}")

        if depth is None:
            self.depth = self.DEPTH
        elif isinstance(depth, int):
            self.depth = depth
        else:
            raise TypeError(f"Атрибут depth ожидает тип int, но получил тип {type(depth)}")

        if start_val_filters is None:
            self.start_val_filters = self.START_VAL_FILTERS
        elif isinstance(start_val_filters, int):
            self.start_val_filters = start_val_filters
        else:
            raise TypeError(f"Атрибут start_val_filters ожидает тип int, но получил тип {type(start_val_filters)}")

        if list_metrics is None:
            self.list_metrics = self.LIST_METRICS
        elif isinstance(list_metrics, list):
            if len(list_metrics) == 0:
                raise TypeError(f"Атрибут list_metrics не должен быть пустым")
            else:
                for fn in list_metrics:
                    if not callable(fn):
                        raise TypeError(f"Атрибут list_metrics ожидает список с вызываемыми функциями!")
                self.list_metrics = list_metrics
        elif callable(list_metrics):
            self.list_metrics = [list_metrics]
        else:
            raise TypeError(f"Атрибут list_metrics ожидает тип Optional[List[Callable]], "
                            f"но получил тип {type(list_metrics)}")

        self.model = self.get_3d_u_net_model(input_shape=(self.input_img_shape + (self.n_channels,)),
                                             n_classes=self.n_classes,
                                             pool_size=self.pool_size,
                                             depth=self.depth,
                                             initial_learning_rate=self.initial_learning_rate,
                                             type_up_convolution=self.type_up_convolution,
                                             start_val_filters=self.start_val_filters,
                                             metrics=self.list_metrics)

    @staticmethod
    def get_3d_u_net_model(input_shape: tuple,
                           n_classes: int,
                           pool_size: Tuple[int, int, int],
                           depth: int,
                           initial_learning_rate: float,
                           type_up_convolution: str,
                           start_val_filters: int,
                           metrics: List[Callable]):
        """
        Возвращает модель 3D-UNet

        Args:
            metrics: Метрики, которые будут вычисляться во время обучения модели (по умолчанию - Мера Сёренсена).
            start_val_filters: Количество фильтров, которые будет иметь первый слой в сети.
            Следующие слои будут содержать число, кратное этому числу.
            depth: глубина сети. Уменьшение глубины может уменьшить объем памяти, необходимый для тренировки.
            input_shape: Форма входных данных (n_channels, x_size, y_size, z_size).
            Размеры x, y и z должны делиться на размер пула в степени глубины UNet, то есть pool_size ^ depth.
            pool_size: Размер пула для максимальных операций объединения.
            n_classes: Количество двоичных меток, которые изучает модель.
            initial_learning_rate: Начальная скорость обучения модели.
            type_up_convolution: тип повышающего слоя. up_sampling_3d использует меньше памяти.
        Returns:
            необученная 3D UNet модель
        """
        inputs = tf.keras.layers.Input(input_shape)
        current_layer = inputs
        levels = []

        # -- Encoder -- #
        for layer_depth in range(depth):
            layer1 = Model3DUnet.get_conv3d(input_layer=current_layer, n_filters=start_val_filters * (2 ** layer_depth))
            layer2 = Model3DUnet.get_conv3d(input_layer=layer1, n_filters=start_val_filters * (2 ** layer_depth) * 2)
            if layer_depth < depth - 1:
                current_layer = tf.keras.layers.MaxPooling3D(pool_size=pool_size)(layer2)
                levels.append([layer1, layer2, current_layer])
            else:
                current_layer = layer2
                levels.append([layer1, layer2])
        # -- Encoder -- #

        # -- Decoder -- #
        for layer_depth in range(depth - 2, -1, -1):
            up_convolution = Model3DUnet.get_up_convolution(pool_size=pool_size,
                                                            type_up_convolution=type_up_convolution,
                                                            n_filters=current_layer.shape[1])(current_layer)
            concat = tf.keras.layers.concatenate([up_convolution, levels[layer_depth][1]], axis=-1)
            current_layer = Model3DUnet.get_conv3d(n_filters=levels[layer_depth][1].shape[1], input_layer=concat)
            current_layer = Model3DUnet.get_conv3d(n_filters=levels[layer_depth][1].shape[1],
                                                   input_layer=current_layer)

        final_convolution = tf.keras.layers.Conv3D(n_classes, (1, 1, 1))(current_layer)
        output = tf.keras.layers.Activation("softmax")(final_convolution)
        model = tf.keras.Model(inputs=inputs, outputs=output)

        if not isinstance(metrics, list):
            metrics = [metrics]

        if n_classes > 1:
            label_wise_dice_metrics = [get_label_dice_coefficient_function(index) for index in range(n_classes)]
            if metrics:
                metrics = metrics + label_wise_dice_metrics
            else:
                metrics = label_wise_dice_metrics

        model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=initial_learning_rate),
                      loss=dice_coefficient_loss, metrics=metrics)
        return model

    @staticmethod
    def get_conv3d(input_layer, n_filters, kernel=(3, 3, 3),
                   padding='same', strides=(1, 1, 1)):
        layer = tf.keras.layers.Conv3D(n_filters, kernel, padding=padding, strides=strides)(input_layer)
        layer = tf.keras.layers.BatchNormalization()(layer)
        try:
            # from keras_contrib.layers.normalization import InstanceNormalization
            # (Azam): Ref.: https://github.com/ellisdg/3DUnetCNN/issues/182
            from keras_contrib.layers.normalization.instancenormalization import InstanceNormalization
        except ImportError:
            raise ImportError("Установите keras_contrib, чтобы использовать нормализацию экземпляра."
                              "\nTry: pip install git+https://www.github.com/keras-team/keras-contrib.git")
        layer = InstanceNormalization()(layer)
        return tf.keras.layers.Activation('relu')(layer)

    @staticmethod
    def get_up_convolution(n_filters, pool_size, kernel_size=(2, 2, 2), strides=(2, 2, 2),
                           type_up_convolution="up_sampling_3d"):
        if type_up_convolution == "conv_3d_transpose":
            return tf.keras.layers.Conv3DTranspose(filters=n_filters, kernel_size=kernel_size,
                                                   strides=strides)
        if type_up_convolution == "up_sampling_3d":
            return tf.keras.layers.UpSampling3D(size=pool_size)

    @staticmethod
    def load_model(path_to_model_file, n_classes):
        print("Loading pre-trained model")
        custom_objects = get_metric_as_costome_object(n_classes)
        return tf.keras.models.load_model(path_to_model_file, custom_objects=custom_objects)

    def update_model_from_file(self, path_to_model_file):
        self.model = self.load_model(path_to_model_file)


if __name__ == "__main__":
    # manage_model = Model3DUnet()
    manager_model = Model3DUnet(input_img_shape=(128, 128, 128,), start_val_filters=16)

    tf.keras.utils.plot_model(manager_model.model, show_shapes=True)
