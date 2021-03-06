"""Всё что связано с получением метрик."""

# TODO Необходимо прибраться в метриках. Сейчас это полигон для тестов
import functools
import numpy as np
import tensorflow as tf
from tensorflow.keras import backend as k


def dice_coefficient(y_true, y_pred,
                     epsilon=0.00001):
    y_true_f = k.flatten(y_true)
    y_pred_f = k.flatten(y_pred)
    intersection = tf.math.reduce_sum(y_true_f * y_pred_f, name='intersection')
    dice_numerator = 2 * intersection + epsilon
    dice_denominator = k.sum(y_true) + k.sum(y_pred) + epsilon
    coefficient = k.mean(dice_numerator / dice_denominator)

    return coefficient


def dice_coef_multilabel(y_true, y_pred, numLabels=5):
    dice = 0
    for index in range(numLabels):
        dice -= dice_coefficient(y_true[..., index], y_pred[..., index])
    return dice


def multiclass_weighted_dice_loss(y_true: tf.Tensor, y_pred: tf.Tensor) -> tf.Tensor:
    """
    Compute weighted Dice loss.
    :param y_true: True masks (tf.Tensor, shape=(<BATCH_SIZE>, <IMAGE_HEIGHT>, <IMAGE_WIDTH>, <N_CLASSES>))
    :param y_pred: Predicted masks (tf.Tensor, shape=(<BATCH_SIZE>, <IMAGE_HEIGHT>, <IMAGE_WIDTH>, <N_CLASSES>))
    :return: Weighted Dice loss (tf.Tensor, shape=(None,))
    """
    # для первого снимка
    #     class_weights = tf.constant([1 / 0.0758, 1 / 0.7865, 1 / 0.1356])
    # # Для первых 295 снимков
    class_weights = tf.constant([1 / 0.166, 1 / 0.6285, 1 / 0.2055])

    axis_to_reduce = range(1, k.ndim(y_pred))  # Reduce all axis but first (batch)
    numerator = y_true * y_pred * class_weights  # Broadcasting
    numerator = 2. * k.sum(numerator, axis=axis_to_reduce)

    denominator = (y_true + y_pred) * class_weights  # Broadcasting
    denominator = k.sum(denominator, axis=axis_to_reduce)

    return 1 - numerator / denominator


def soft_dice_loss(y_true, y_pred,
                   axis: tuple = (0, 1, 2),
                   epsilon: float = 0.00001):
    """
    Вычислите среднюю потерю по всем классам аномалий.
    Args:
        y_true (Tensorflow tensor): тензор истинностных значений для всех классов.
                                    shape: (z_dim, x_dim, y_dim, num_classes)
        y_pred (Tensorflow tensor): тензор мягких предсказаний для всех классов.
                                    shape: (z_dim, x_dim, y_dim, num_classes)
        axis (tuple): пространственные оси для суммирования.
        epsilon (float): к числителю и знаменателю добавлена
        небольшая константа, чтобы избежать ошибок деления на 0.
    Returns:
        dice_loss (float): computed value of dice loss.
    """

    dice_numerator = 2. * k.sum(y_true * y_pred, axis=axis) + epsilon
    dice_denominator = k.sum(y_true ** 2, axis=axis) + k.sum(y_pred ** 2, axis=axis) + epsilon
    dice_loss = 1 - k.mean(dice_numerator / dice_denominator)

    return dice_loss


def dice_coefficient_loss(y_true, y_pred, axis=-1,
                          epsilon=0.00001):
    dice_numerator = 2 * k.sum(y_true * y_pred, axis=axis) + epsilon
    dice_denominator = k.sum(y_true ** 2, axis=axis) + k.sum(y_pred ** 2, axis=axis) + epsilon
    dice_loss = 1 - k.mean(dice_numerator / dice_denominator)
    return dice_loss


def label_wise_dice_coefficient(y_true, y_pred, label_index):
    return dice_coefficient(y_true[..., label_index], y_pred[..., label_index])


def get_label_dice_coefficient_function(label_index):
    f = functools.partial(label_wise_dice_coefficient, label_index=label_index)
    f.__setattr__('__name__', 'label_{0}_dice_coef'.format(label_index))
    return f


def get_dice_metric_as_custom_object(count):
    d = {}
    for index in range(count):
        f = get_label_dice_coefficient_function(index)
        d[f.__name__] = f
    d['dice_coefficient_loss'] = dice_coefficient_loss
    d['dice_coefficient'] = dice_coefficient

    return d


def jaccard_coefficient(y_true, y_pred, smooth: float = 1e-2, pull: float = 0.0):
    """ Коэффициент Жаккара для tensorflow. """

    y_true_f = k.cast_to_floatx(y_true)
    y_pred_f = k.cast_to_floatx(y_pred)
    smooth_fact = pull * smooth
    intersection = k.sum(y_true_f * y_pred_f, axis=-1)
    union = k.sum(y_true_f, axis=-1) + k.sum(y_pred_f, axis=-1) - intersection
    return (intersection + smooth_fact) / (union + smooth)


def jaccard_distance(y_true, y_pred,
                     smooth: float = 1e-2,
                     pull: float = 0.0):
    smooth_fract = pull * smooth
    y_true_f = k.cast_to_floatx(y_true)
    y_pred = k.cast_to_floatx(y_pred)
    if len(y_true_f.shape) == 5:
        axis = (-4, -3, -2)
    elif len(y_true_f.shape) == 4:
        axis = (-3, -2)
    else:
        axis = -2
    intersection = k.sum(y_true_f * y_pred, axis=axis)
    union = k.sum(y_true_f, axis=axis) + k.sum(y_pred, axis=axis) - intersection
    jac = (intersection + smooth_fract) / (union + smooth)
    jac = k.mean(jac, axis=-1)

    return 1.0 - jac


def label_wise_jaccard_coefficient(y_true, y_pred, label_index):
    return jaccard_coefficient(y_true[..., label_index], y_pred[..., label_index])


def get_label_jaccard_coefficient_function(label_index):
    f = functools.partial(label_wise_jaccard_coefficient, label_index=label_index)
    f.__setattr__('__name__', 'label_{0}_jaccard_coef'.format(label_index))
    return f


# custom
def get_jaccard_metric_as_custom_object(count):
    d = {}
    for index in range(count):
        f = get_label_jaccard_coefficient_function(index)
        d[f.__name__] = f
    d['jaccard_distance'] = jaccard_distance
    d['jaccard_coefficient'] = jaccard_coefficient

    return d


if __name__ == "__main__":
    # TEST CASES

    pred = np.ones((10, 15, 15, 15, 3))
    label = pred
    label[:, :, :, :, 0] = 0
    label[:, :, :, :, 1] = 0.5

    jc = jaccard_distance(label, pred)
