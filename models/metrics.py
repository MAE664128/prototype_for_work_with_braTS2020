"""Всё что связано с получением метрик."""

# TODO Необходимо прибраться в метриках. Сейчас это полигон для тестов
import functools
from tensorflow.keras import backend as k
import tensorflow as tf
import numpy as np


def dice_coefficient_old(y_true, y_pred, smooth=1.):
    """
    Мера Сёренсена — бинарная мера сходства.
    """
    y_true_f = k.flatten(y_true)
    y_pred_f = k.flatten(y_pred)
    intersection = k.sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (k.sum(y_true_f) + k.sum(y_pred_f) + smooth)


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


def soft_dice_loss(y_true: tf.Tensor, y_pred: tf.Tensor,
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


def jaccard_coefficient(y_true: tf.Tensor, y_pred: tf.Tensor, smooth: float = 1e-2, pull: float = 0.0):
    """ Коэффициент Жаккара для tensorflow. """

    y_true_f = k.flatten(y_true)
    y_pred_f = k.flatten(y_pred)
    # y_true_f = k.cast_to_floatx(y_true)
    smooth_fact = pull * smooth
    intersection = k.sum(y_true_f * y_pred_f)
    union = k.sum(y_true_f) + k.sum(y_pred_f) - intersection
    return (intersection + smooth_fact) / (union + smooth)


def jaccard_distance(y_true: tf.Tensor, y_pred: tf.Tensor,
                     smooth: float = 1e-2,
                     pull: float = 0.0,
                     limit: float = 1e3):

    weights = [1, 1, 1]
    smooth_fract = pull * smooth
    ofs = 1.0 / limit
    y_true_f = k.cast_to_floatx(y_true)
    list_jac = []
    for i in range(3):
        intersection = k.sum(y_true_f[..., i] * y_pred[..., i])
        union = k.sum(y_true_f[..., i]) + k.sum(y_pred[..., i]) - intersection
        jac = (intersection + smooth_fract) / (union + smooth)
        list_jac.append(jac / weights[i])

    jac = k.mean(tf.convert_to_tensor(list_jac, dtype=tf.float32))

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
    pred = np.expand_dims(np.expand_dims(np.eye(2), 0), -1)
    label = np.expand_dims(np.expand_dims(np.array([[1.0, 1.0], [0.0, 0.0]]), 0), -1)

    print("Test Case #1")
    print("pred:")
    print(pred[0, :, :, 0])
    print("label:")
    print(label[0, :, :, 0])

    dc = dice_coefficient(label, pred, epsilon=1)
    print(f"dice coefficient: {dc.numpy():.4f}")
    jc = jaccard_coefficient(label, pred)
    print(f"jaccard coefficient: {jc.numpy():.4f}")

    print("\n")

    print("Test Case #2")
    pred = np.expand_dims(np.expand_dims(np.eye(2), 0), -1)
    label = np.expand_dims(np.expand_dims(np.array([[1.0, 1.0], [0.0, 1.0]]), 0), -1)

    print("pred:")
    print(pred[0, :, :, 0])
    print("label:")
    print(label[0, :, :, 0])

    dc = dice_coefficient(pred, label, epsilon=1)
    print(f"dice coefficient: {dc.numpy():.4f}")
    jc = jaccard_coefficient(label, pred)
    print(f"jaccard coefficient: {jc.numpy():.4f}")
    print("\n")

    print("Test Case #3")
    pred = np.zeros((2, 2, 2, 1))
    pred[0, :, :, :] = np.expand_dims(np.eye(2), -1)
    pred[1, :, :, :] = np.expand_dims(np.eye(2), -1)

    label = np.zeros((2, 2, 2, 1))
    label[0, :, :, :] = np.expand_dims(np.array([[1.0, 1.0], [0.0, 0.0]]), -1)
    label[1, :, :, :] = np.expand_dims(np.array([[1.0, 1.0], [0.0, 1.0]]), -1)

    print("pred:")
    print("class = 0")
    print(pred[0, :, :, 0])
    print("class = 1")
    print(pred[1, :, :, 0])
    print("label:")
    print("class = 0")
    print(label[0, :, :, 0])
    print("class = 1")
    print(label[1, :, :, 0])

    dc = dice_coefficient(pred, label, epsilon=1)
    print(f"dice coefficient: {dc.numpy():.4f}")
    jc = jaccard_coefficient(label, pred)
    print(f"jaccard coefficient: {jc.numpy():.4f}")

    print("\n")

    print("Test Case #4")
    pred = np.zeros((2, 2, 2, 1))
    pred[0, :, :, :] = np.expand_dims(np.eye(2), -1)
    pred[1, :, :, :] = np.expand_dims(np.eye(2), -1)

    label = pred
    label[0, 0, 0, 0] = 0

    print("pred:")
    print("class = 0")
    print(pred[0, :, :, 0])
    print("class = 1")
    print(pred[1, :, :, 0])
    print("label:")
    print("class = 0")
    print(label[0, :, :, 0])
    print("class = 1")
    print(label[1, :, :, 0])

    dc = dice_coefficient(pred, label, epsilon=1)
    print(f"dice coefficient: {dc.numpy():.4f}")
    jc = jaccard_coefficient(label, pred)
    print(f"jaccard coefficient: {jc.numpy():.4f}")