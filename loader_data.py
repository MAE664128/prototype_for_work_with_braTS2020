from __future__ import annotations

import os
import random
from typing import Optional, List, Tuple, Dict, Union

import numpy as np
import SimpleITK as s_itk
import matplotlib.pyplot as plt
from IPython.display import HTML
from matplotlib import animation

plt.rcParams['animation.ffmpeg_path'] = 'C:/FFmpeg/bin/ffmpeg.exe'


class AnimateView:
    def __init__(self):
        self.fig, self.ax = plt.subplots()
        self.ims = []
        self.ani = None

    def get_ani(self, img, min_val=None, max_val=None):
        if min_val is None:
            min_val = 0
        if max_val is None:
            max_val = img.max()
        z, x, y = img.shape
        self.ims = []
        for ind_z in range(z):
            im = self.ax.imshow(img[ind_z, :, :], animated=True, cmap='gray', vmin=min_val, vmax=max_val)

            self.ims.append([im])
        self.ani = animation.ArtistAnimation(self.fig,
                                             self.ims,
                                             interval=50,
                                             blit=True)
        return self.ani

    @staticmethod
    def convert_to_html(ani):
        return HTML(ani.to_html5_video())


# animate1 = AnimateView()


def gray2rgb(gray):
    # gray it is 3d array
    z, x, y = gray.shape
    tmp_gray = np.interp(gray, (gray.min(), gray.max()), (0, 254))
    tmp_gray = tmp_gray.astype(np.uint8)
    img_new = np.zeros((z, x, y, 3))
    img_new[:, :, :, 0] = tmp_gray
    img_new[:, :, :, 1] = tmp_gray
    img_new[:, :, :, 2] = tmp_gray
    return img_new


class PreprocessLoadData:
    __PATH_TO_DATA = f"data/BraTS2020_TrainingData/MICCAI_BraTS2020_TrainingData"
    __PATH_TO_CACHE = f"cache/"
    __PATH_TO_BATCH_CACHE = f"batch_cache/"
    __FILENAME_POSTFIX = ["_t1", "_t2", "_t1ce"]
    __SEG_NAME_POSTFIX = "_seg"
    __TYPE_DATA = "TrainingData"
    __N_CLASSES = 3
    # Размер вокселя в изображении к которому нужно привести все входные данные
    __SPACING = (1.0, 1.0, 1.0)
    # Размер изображения к которому необходимо привести все входные данные
    __SHAPE = (256, 256, 192)
    # __KERNEL_SIZE - Размер ядра для генератора
    __KERNEL_SIZE = (64, 64, 64)
    # __STEP - Шаг скользящего окна
    __STEP = (32, 32, 32)
    # __BATH_SIZE - Количество кусков которые возвращает генератор
    __BATH_SIZE = 1

    def __init__(self,
                 path_to_data: Optional[str] = None,
                 filename_postfix: Optional[List[str]] = None,
                 seg_name_postfix: Optional[str] = None,
                 type_data: Optional[str] = None,
                 kernel_size: Optional[Union[Tuple[int, int, int], Tuple[int, int]]] = None,
                 step: Optional[Union[Tuple[int, int, int], Tuple[int, int]]] = None,
                 batch_size: Optional[int] = None
                 ):

        self.__batch_size: int
        self.__step: Union[Tuple[int, int, int], Tuple[int, int]]
        self.__kernel_size: Union[Tuple[int, int, int], Tuple[int, int]]
        self.__path_to_data: Optional[str]
        self.__filename_postfix: Optional[List[str]]
        self.__seg_name_postfix: Optional[str]
        self.__type_data: Optional[str]
        self.__file_paths: Optional[List[dict]]

        self.__batch_size = self.__BATH_SIZE if batch_size is None else batch_size
        self.__step = self.__STEP if step is None else step
        self.__kernel_size = self.__KERNEL_SIZE if kernel_size is None else kernel_size
        self.__path_to_data = self.__PATH_TO_DATA if path_to_data is None else path_to_data
        self.__filename_postfix = self.__FILENAME_POSTFIX if filename_postfix is None else filename_postfix
        self.__seg_name_postfix = self.__SEG_NAME_POSTFIX if seg_name_postfix is None else seg_name_postfix
        self.__type_data = self.__TYPE_DATA if type_data is None else type_data
        self.__file_paths = None

        self.valid_params()

    def length_data(self):
        return 0 if self.__file_paths is None else len(self.__file_paths)

    def need_to_find_seg(self) -> bool:
        return self.__type_data in ["TrainingData", "TestingData"]

    @property
    def file_paths(self):
        return self.__file_paths

    @property
    def step(self):
        return self.__step

    @step.setter
    def step(self, kernel_size: Tuple[int, int, int] or Tuple[int, int]):
        self.__step = kernel_size
        self.valid_params()

    @property
    def kernel_size(self):
        return self.__kernel_size

    @kernel_size.setter
    def kernel_size(self, kernel_size: Tuple[int, int, int] or Tuple[int, int]):
        self.__kernel_size = kernel_size
        self.valid_params()

    @property
    def type_data(self):
        return self.__type_data

    @type_data.setter
    def type_data(self, new_type_data: str):
        self.__type_data = new_type_data
        self.valid_params()

    @property
    def seg_name_postfix(self):
        return self.__seg_name_postfix

    @seg_name_postfix.setter
    def seg_name_postfix(self, seg_name_postfix: str):
        self.__seg_name_postfix = seg_name_postfix
        self.valid_params()

    @property
    def filename_postfix(self):
        return self.__filename_postfix

    @filename_postfix.setter
    def filename_postfix(self, filename_postfix: List[str]):
        self.__filename_postfix = filename_postfix
        self.valid_params()

    @property
    def path_to_data(self):
        return self.__path_to_data

    @path_to_data.setter
    def path_to_data(self, path_to_train_data: str):
        self.__path_to_data = path_to_train_data
        self.valid_params()

    def valid_params(self):
        # path_to_dir_with_data_for_train
        assert isinstance(self.__path_to_data, str), \
            f"[PreprocessLoadData]: Параметр path_to_train_data не задан или имеет тип отличный от str"
        assert not os.path.isfile(
            self.__path_to_data), f"[PreprocessLoadData]: Директория не найдена {self.__path_to_data}"

        # filename_postfix
        assert isinstance(self.__filename_postfix, list), \
            f"[PreprocessLoadData]: filename_postfix ожидает тип Optional[List[str]], " \
            f"но получил тип {type(self.__filename_postfix)}"
        assert len(
            self.__filename_postfix) != 0, \
            f"[PreprocessLoadData]: filename_postfix ожидает не пустой список, но получил []"
        assert isinstance(self.__filename_postfix[0], str), \
            f"[PreprocessLoadData]: filename_postfix ожидает тип Optional[List[str]], " \
            f"но получил тип List{[type(self.__filename_postfix[0])]}"

        # default_seg_name_postfix
        if self.__seg_name_postfix is not None:
            assert isinstance(self.__seg_name_postfix, str), \
                f"[PreprocessLoadData]: seg_name_postfix ожидает тип Optional[str], " \
                f"но получил тип {type(self.__seg_name_postfix)}"

        # type_data
        assert isinstance(self.__type_data, str), f"[PreprocessLoadData]: type_data ожидает тип Optional[str], " \
                                                  f"но получил тип {type(self.__type_data)}"

        # kernel_size
        assert isinstance(self.__kernel_size, tuple), \
            f"[PreprocessLoadData]: kernel_size ожидает тип Optional[tuple], " \
            f"но получил тип {type(self.__kernel_size)}"
        if isinstance(self.__kernel_size, tuple):
            assert len(self.__kernel_size) in [2, 3], \
                f"[PreprocessLoadData]: kernel_size ожидает tuple размерностью 2 или 3, " \
                f"но получил tuple размером {len(self.__kernel_size)}"

        # step
        assert isinstance(self.__step, tuple), \
            f"[PreprocessLoadData]: step ожидает тип Optional[tuple], " \
            f"но получил тип {type(self.__step)}"
        if isinstance(self.__step, tuple):
            assert len(self.__step) in [2, 3], \
                f"[PreprocessLoadData]: kernel_size ожидает tuple размерностью 2 или 3, " \
                f"но получил tuple размером {len(self.__step)}"

        # kernel_size and step
        assert len(self.__kernel_size) == len(self.__step), \
            f"[PreprocessLoadData]: kernel_size и step должны иметь одинаковы размер. " \
            f"len(kernel_size)={len(self.__kernel_size)}\tlen(step)={len(self.__step)}"

        # __batch_size
        assert isinstance(self.__batch_size, int), f"[PreprocessLoadData]: batch_size ожидает тип int, " \
                                                   f"но получил тип {type(self.__batch_size)}"

    def find_files(self) -> PreprocessLoadData:
        """
        Проверяет наличие файлов в директории
        """
        result_list = []
        for folder, dirs, files in os.walk(self.__path_to_data):
            if dirs == [] and len(files) >= len(self.__filename_postfix):
                d = {}
                for file in files:
                    for postfix in self.__filename_postfix:
                        if postfix not in d:
                            if file.find(postfix) != -1:
                                d[postfix] = os.path.join(folder, file)
                    if self.need_to_find_seg() and self.__seg_name_postfix not in d:
                        if file.find(self.__seg_name_postfix) != -1:
                            d[self.__seg_name_postfix] = os.path.join(folder, file)

                for postfix in self.__filename_postfix:
                    assert postfix in d, f"Отсутствует файл {postfix} в директории:\n\t {folder}"
                if self.need_to_find_seg():
                    assert self.__seg_name_postfix in d, \
                        f"Отсутствует файл сегментации в директории:\n\t {folder}"
                result_list.append(d)
        self.__file_paths = result_list if result_list != [] else None
        return self

    def load_and_preprocess_nii(self, file_paths: list) -> Dict[str, s_itk.Image]:
        """
        Загружаем в память медицинские изображения из списка file_paths,
        масштабируем и возвращаем dict содержащий масштабированные изображения в формате SimpleITK.
        """
        for ind, dict_file_paths in enumerate(file_paths):
            d = {}
            for key, val in dict_file_paths.items():
                # Формируем путь к файлу с уже масштабированными изображениями (к кешу)
                path_to_cache_file = f"{self.__PATH_TO_CACHE}{os.path.split(val)[1]}"
                # Проверяем есть ли в папке кеш масштабированное изображение. Если да, то подгружаем его.
                if os.path.exists(path_to_cache_file):
                    d[key] = s_itk.ReadImage(path_to_cache_file)
                else:
                    d[key] = s_itk.ReadImage(val)
                    # Если размер изображения не соответствует требованиям, то изменяем его
                    if d[key].GetSpacing() != self.__SPACING or s_itk.GetArrayFromImage(d[key]).shape != self.__SHAPE:
                        d[key] = resample_img(d[key], out_spacing=self.__SPACING, out_size=self.__SHAPE,
                                              is_label=key == self.__SEG_NAME_POSTFIX)
                        # Сохраняем масштабированное изображение в папку кеш
                        writer = s_itk.ImageFileWriter()
                        writer.SetFileName(path_to_cache_file)
                        writer.Execute(d[key])
            yield d

    def take_img_as_numpy(self,
                          file_paths: list,
                          required_postfixes: Optional[List[str]] = None) -> np.ndarray and Optional[np.ndarray]:
        """
        Функция генератор возвращает изображение и маску из набора.
        Если маска для изображения отсутствует, то будет возвращено None

        :param:

            required_postfixes - Список префиксов указывающий какие данные необходимо вернуть
            file_paths - Список файлов для которых нужно вернуть фрагментны

        """
        if required_postfixes is None:
            required_postfixes = self.__FILENAME_POSTFIX
        else:
            for postfix in required_postfixes:
                assert postfix in self.__FILENAME_POSTFIX, \
                    f"[PreprocessLoadData]: Параметр required_postfixes содержит постфикс отсутствующий в данных"
        d: dict
        for d in self.load_and_preprocess_nii(file_paths=file_paths):
            l_img = []
            for postfix in required_postfixes:
                l_img.append(s_itk.GetArrayFromImage(d[postfix]))
            images = np.stack(l_img, axis=-1).astype(np.float32)
            mask = s_itk.GetArrayFromImage(d[self.__seg_name_postfix]) if self.__seg_name_postfix in d else None
            yield images, mask

    def get_generator_data(self,
                           type_g: str,
                           required_postfixes: Optional[List[str]] = None,
                           is_whitening: bool = True,
                           threshold: float = 0.05,
                           random_change_plane: bool = False,
                           augmentation: bool = False):
        """
        Args:
            type_g:
            required_postfixes:
            is_whitening:
            random_change_plane:
            augmentation:
            threshold: интервал присутствия мозга на изображении (по умолчанию мозг занимает не менее 5%)


        Returns:

        """
        for img, msk in self.get_kernel_generator_data(type_g,
                                                       required_postfixes,
                                                       random_change_plane,
                                                       is_whitening,
                                                       threshold,
                                                       augmentation):
            yield img, msk

    def get_kernel_generator_data(self,
                                  type_g: str,
                                  required_postfixes: Optional[List[str]] = None,
                                  random_change_plane: bool = False,
                                  is_whitening: bool = True,
                                  threshold: float = 0.05,
                                  augmentation: bool = False):
        """ Возвращает 2D или 3D генератор в зависимости от размерности kernel """
        if required_postfixes is None:
            required_postfixes = ["_t1", "_t1ce", "_t2"]
        data_length = self.length_data()
        train_length = round(data_length * 0.8)
        train_file_paths: list = self.__file_paths[0:train_length]

        # TODO УБРАТЬ
        #   Добавлено для отладки, что бы обучать только по 1 снимку
        train_file_paths: list = self.__file_paths[0:8]

        test_file_paths: list = self.__file_paths[train_length:data_length - 1]
        if type_g == "train":
            return self.__get_kernel_generator_data(file_paths=train_file_paths, type_g=type_g,
                                                    required_postfixes=required_postfixes,
                                                    random_change_plane=random_change_plane,
                                                    is_whitening=is_whitening,
                                                    threshold=threshold,
                                                    augmentation=augmentation)
        elif type_g == "test":
            return self.__get_kernel_generator_data(file_paths=test_file_paths, type_g=type_g,
                                                    required_postfixes=required_postfixes,
                                                    random_change_plane=random_change_plane,
                                                    is_whitening=is_whitening,
                                                    threshold=threshold, augmentation=augmentation)
        else:
            assert False, \
                f"[PreprocessLoadData]: Параметр type_g ожидает значение \"train\" или \"test\", но получил {type_g}"

    def __get_kernel_generator_data(self, file_paths: list, type_g: str,
                                    required_postfixes: List[str],
                                    random_change_plane: bool = False,
                                    is_whitening: bool = True,
                                    threshold: float = 0.05,
                                    augmentation: bool = False):

        tmp_list_img = []
        tmp_list_msk = []

        while True:
            if type_g == "train":
                random.shuffle(file_paths)
            for input_image, input_mask in self.take_img_as_numpy(required_postfixes=required_postfixes,
                                                                  file_paths=file_paths):
                assert input_mask is not None, "Маска не найдена"
                input_image = input_image / 255.0
                if is_whitening:
                    for c in range(input_image.shape[-1]):
                        input_image[..., c] = whitening(input_image[..., c])

                input_mask = split_mask_values_by_brats_channel(input_mask)

                if random_change_plane:
                    # TODO Переписать на универсальную, независящую от размеров функцию
                    #  реконструкции плоскостей. Сейчас функция работает только для 3d с одинаковыми сторонами
                    # Только для 3д
                    id_plane = random.randint(1, 3)
                    if id_plane == 1:
                        # Сагиттальная плоскость
                        input_image = np.transpose(input_image, [2, 1, 0, 3])
                        input_mask = np.transpose(input_mask, [2, 1, 0, 3])
                        input_image = np.rot90(input_image, 1, axes=(1, 2))
                        input_mask = np.rot90(input_mask, 1, axes=(1, 2))
                    elif id_plane == 2:
                        # Корональная
                        input_image = np.transpose(input_image, [1, 0, 2, 3])
                        input_mask = np.transpose(input_mask, [1, 0, 2, 3])
                        input_image = np.rot90(input_image, 2, axes=(1, 2))
                        input_mask = np.rot90(input_mask, 2, axes=(1, 2))

                img, msk = self.__kernel_split(input_image, input_mask)
                i_ind, j_ind, k_ind, _, _, _, _ = img.shape
                for i in range(i_ind):
                    for j in range(j_ind):
                        for k in range(k_ind):
                            if len(tmp_list_img) >= self.__batch_size:
                                tmp_list = list(zip(tmp_list_img, tmp_list_msk))
                                random.shuffle(tmp_list)
                                tmp_list_img, tmp_list_msk = zip(*tmp_list)
                                del tmp_list
                                result_img = np.asarray(tmp_list_img, dtype=np.float32).reshape(
                                    (self.__batch_size,) + self.__kernel_size + (img.shape[-1],))
                                result_msk = np.asarray(tmp_list_msk, dtype=np.float32).reshape(
                                    (self.__batch_size,) + self.__kernel_size + (msk.shape[-1],))
                                tmp_list_img = []
                                tmp_list_msk = []

                                yield result_img, result_msk
                            else:
                                mean_bg = img[i, j, k, :, :, :, 1].min() / img[i, j, k, :, :, :, 1].size
                                mean_bg = np.count_nonzero(img[i, j, k, :, :, :, 1] != mean_bg)
                                if mean_bg < threshold:
                                    continue

                                if augmentation:
                                    if random.random() < 0.5:
                                        res_flip = flip_or_rot_img(img[i, j, k, :, :, :, :], msk[i, j, k, :, :, :, :])
                                        tmp_list_img.append(res_flip[0])
                                        tmp_list_msk.append(res_flip[1])
                                else:
                                    tmp_list_img.append(img[i, j, k, :, :, :, :])
                                    tmp_list_msk.append(msk[i, j, k, :, :, :, :])

    def __kernel_split(self, image_array: np.ndarray,
                       mask_array: np.ndarray,
                       kernel_size: Optional[Tuple[int, int, int]] = None,
                       step: Optional[Tuple[int, int, int]] = None):
        """Разделение 3д-массива на блоки размером соответствующим размеру kernel_size.
        Блоки формируются по принципу скользящего окна c шагом step.

        Args:
            image_array: многомерный массив с изображениями.
            mask_array: маска изображения.
            kernel_size: размер скользящего окна
            step: шаг скользящего окна
        Returns:
            x_piece, y_piece
        """
        kernel_n: tuple = kernel_size
        step_n: tuple = step

        if kernel_size is None:
            kernel_size = self.__kernel_size
        if step is None:
            step = self.__step
        if len(kernel_size) == 2:
            kernel_n = (1,) + kernel_size
            step_n = (1,) + step
        if len(kernel_size) == 3:
            kernel_n = kernel_size
            step_n = step
        img = roll_3d_over_4d_array(image_array, kernel_n, step_n)
        msk = roll_3d_over_4d_array(mask_array, kernel_n, step_n)
        return img, msk


def split_mask_values_by_channel(y_true, n_channel):
    new_y_true = np.zeros(y_true.shape + (n_channel,), dtype=np.float32)

    for i in range(n_channel):
        new_y_true[..., i] = np.where(np.logical_and(y_true <= i, y_true > i - 1), 1, 0)
    return new_y_true


def split_mask_values_by_brats_channel(y_true):
    new_y_true = np.zeros(y_true.shape + (3,), dtype=np.float32)
    # NCR/NET
    new_y_true[..., 0] = np.where(np.logical_and(y_true <= 1, y_true > 0), 1, 0)
    # peritumoral edema
    new_y_true[..., 1] = np.where(np.logical_and(y_true <= 2, y_true > 1), 1, 0)
    # GD-enhancing tumor
    new_y_true[..., 2] = np.where(np.logical_and(y_true <= 4, y_true > 3), 1, 0)

    return new_y_true


# borrowed from DLTK
def whitening(image):
    """Отбеливание. Нормализует изображение до нулевого среднего и единичной дисперсии."""
    image = image.astype(np.float32)
    mean = np.mean(image)
    std = np.std(image)
    if std > 0:
        ret = (image - mean) / std
    else:
        ret = image * 0.
    return ret


def flip_or_rot_img(img, msk=None):
    """
    Args:
        img, msk: массив размерностью (bath_size, z, x, y, channel) или  (bath_size, x, y, channel)

    Returns: перевернутый случайным образом массив
    """
    dim = len(img.shape)
    i_ax = random.randint(1, dim - 2)
    if random.random() < 0.5:
        img = np.flip(img, axis=i_ax)
        if msk is not None:
            msk = np.flip(msk, axis=i_ax)
    else:
        img = np.rot90(img, i_ax, axes=(dim - 3, dim - 2))
        if msk is not None:
            msk = np.rot90(msk, i_ax, axes=(dim - 3, dim - 2))
    return img, msk


def resample_img(itk_image,
                 out_spacing: Optional[Tuple[float, float, float]] = None,
                 out_size: Optional[Tuple[int, int, int]] = None,
                 is_label: bool = False):
    # Измените разрешение изображений с помощью SimpleITK
    if out_spacing is None:
        out_spacing = (1.0, 1.0, 1.0)
    original_spacing = itk_image.GetSpacing()
    original_size = itk_image.GetSize()

    if out_size is None:
        out_size = (
            int(np.round(original_size[0] * (original_spacing[0] / out_spacing[0]))),
            int(np.round(original_size[1] * (original_spacing[1] / out_spacing[1]))),
            int(np.round(original_size[2] * (original_spacing[2] / out_spacing[2]))))

    resample = s_itk.ResampleImageFilter()
    resample.SetOutputSpacing(out_spacing)
    resample.SetSize(out_size)
    resample.SetOutputDirection(itk_image.GetDirection())
    resample.SetOutputOrigin(itk_image.GetOrigin())
    resample.SetTransform(s_itk.Transform())
    resample.SetDefaultPixelValue(0)

    if is_label:
        resample.SetInterpolator(s_itk.sitkNearestNeighbor)
    else:
        resample.SetInterpolator(s_itk.sitkBSpline)

    return resample.Execute(itk_image)


# модификация кода заимствованного из статьи https://habr.com/ru/post/489734/
def roll_3d_over_4d_array(nd_array: np.ndarray, kernel_shape: Tuple[int, int, int],
                          step: Tuple[int, int, int]) -> np.ndarray:
    """3х мерное окно для 4х мерного массива.

        Args:
            nd_array: 3х мерный массив.
            kernel_shape: размер M мерного окна.
            step: Шаг движения окна по осям (dz, dy, dx), где
                dx - шаг по горизонтали, абсцисса, количество столбцов;
                dy - шаг по вертикали, ордината, количество строк;
                dz - поперечный шаг, аппликат, количество слоев.
        Returns:
            Массив под_массивов находящихся в исходном массиве
        """
    shape = ((nd_array.shape[-4] - kernel_shape[-3]) // step[-3] + 1,)
    shape = shape + ((nd_array.shape[-3] - kernel_shape[-2]) // step[-2] + 1,)
    shape = shape + ((nd_array.shape[-2] - kernel_shape[-1]) // step[-1] + 1,)
    shape = shape + kernel_shape + (nd_array.shape[-1],)

    strides = (nd_array.strides[-4] * step[-3],)
    strides = strides + (nd_array.strides[-3] * step[-2],)
    strides = strides + (nd_array.strides[-2] * step[-1],)
    strides = strides + nd_array.strides[-4:]
    return np.lib.stride_tricks.as_strided(nd_array, shape=shape, strides=strides)


if __name__ == "__main__":
    pass
