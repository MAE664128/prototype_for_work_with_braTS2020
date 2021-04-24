"""Всё что связано с получением метрик."""
import functools
from tensorflow.keras import backend as k


def dice_coefficient(y_true, y_pred, smooth=1.):
    """
    Мера Сёренсена — бинарная мера сходства.
    """
    y_true_f = k.flatten(y_true)
    y_pred_f = k.flatten(y_pred)
    intersection = k.sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (k.sum(y_true_f) + k.sum(y_pred_f) + smooth)


def dice_coefficient_loss(y_true, y_pred):
    return -dice_coefficient(y_true, y_pred)


def weighted_dice_coefficient(y_true, y_pred, axis=(-3, -2, -1), smooth=0.00001):
    """
    Взвешенный Мера Сёренсена. Ось по умолчанию предполагает структуру данных "сначала каналы".
    """
    return k.mean(2. * (k.sum(y_true * y_pred, axis=axis) + smooth / 2) / (
            k.sum(y_true, axis=axis) + k.sum(y_pred, axis=axis) + smooth))


def weighted_dice_coefficient_loss(y_true, y_pred):
    return -weighted_dice_coefficient(y_true, y_pred)


def label_wise_dice_coefficient(y_true, y_pred, label_index):
    return dice_coefficient(y_true[:, label_index], y_pred[:, label_index])


def get_label_dice_coefficient_function(label_index):
    f = functools.partial(label_wise_dice_coefficient, label_index=label_index)
    f.__setattr__('__name__', 'label_{0}_dice_coef'.format(label_index))
    return f

def get_metric_as_costome_object(count):
    d = {}
    for index in range(count):
        f = get_label_dice_coefficient_function(index)
        d[f.__name__] = f
    d['dice_coefficient_loss'] = dice_coefficient_loss
    d['dice_coefficient'] = dice_coefficient
    d['weighted_dice_coefficient'] = weighted_dice_coefficient
    d['weighted_dice_coefficient_loss']: weighted_dice_coefficient_loss

    return d