from __future__ import annotations

import gc
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

    def get_ani(self, img):
        z, x, y = img.shape
        max_val = img.max()
        self.ims = []
        for ind_z in range(z):
            im = self.ax.imshow(img[ind_z, :, :], animated=True, cmap='gray', vmin=0, vmax=max_val)
            if ind_z == 0:
                self.ax.imshow(img[ind_z, :, :], cmap='gray', vmin=0, vmax=255)
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

    __PATH_TO_DATA = f"C:/Users/Alexsandr/Desktop/brain-gliom/data/BraTS2020_TrainingData/MICCAI_BraTS2020_TrainingData"
    __PATH_TO_CACHE = f"C:/Users/Alexsandr/Desktop/brain-gliom/cache/"
    __FILENAME_POSTFIX = ["_t1", "_t2", "_t1ce", "_flair"]
    __SEG_NAME_POSTFIX = "_seg"
    __TYPE_DATA = "TrainingData"
    __N_CLASSES = 5
    # Размер вокселя в изображении к которому нужно привести все входные данные
    __SPACING = (1.0, 1.0, 1.0)
    # Размер изображения к которому необходимо привести все входные данные
    __SHAPE = (256, 256, 160)
    # __KERNEL_SIZE - Размер ядра для генератора
    __KERNEL_SIZE = (64, 64, 64)
    # __STEP - Шаг скользящего окна
    __STEP = (10, 10, 10)
    # __BATH_SIZE - Количество кусков которые возвращает генератор
    __BATH_SIZE = 3

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
    def type_data(self, type_data: str):
        self.__type_data = type_data
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

        for ind, dict_file_paths in enumerate(file_paths):
            d = {}
            for key, val in dict_file_paths.items():
                path_to_cache_file = f"{self.__PATH_TO_CACHE}{os.path.split(val)[1]}"
                if os.path.exists(path_to_cache_file):
                    d[key] = s_itk.ReadImage(path_to_cache_file)
                else:
                    d[key] = s_itk.ReadImage(val)
                    if d[key].GetSpacing() != self.__SPACING or s_itk.GetArrayFromImage(d[key]).shape != self.__SHAPE:
                        d[key] = resample_img(d[key], out_spacing=self.__SPACING, out_size=self.__SHAPE,
                                              is_label=key == self.__SEG_NAME_POSTFIX)
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

    def get_kernel_generator_data(self,
                                  type_g: str,
                                  required_postfixes: Optional[List[str]] = None):
        """ Возвращает 2D или 3D генератор в зависимости от размерности kernel """
        if required_postfixes is None:
            required_postfixes = ["_t1", "_t1ce", "_t2"]
        data_length = self.length_data()
        train_length = round(data_length * 0.8)
        train_file_paths: list = self.__file_paths[0:train_length]
        test_file_paths: list = self.__file_paths[train_length:data_length - 1]
        if type_g == "train":
            return self.__get_kernel_generator_data(file_paths=train_file_paths, type_g=type_g,
                                                    required_postfixes=required_postfixes)
        elif type_g == "test":
            return self.__get_kernel_generator_data(file_paths=test_file_paths, type_g=type_g,
                                                    required_postfixes=required_postfixes)
        else:
            assert False, \
                f"[PreprocessLoadData]: Параметр type_g ожидает значение \"train\" или \"test\", но получил {type_g}"

    def __get_kernel_generator_data(self, file_paths: list, type_g: str,
                                    required_postfixes: List[str]):
        while True:
            if type_g == "train":
                random.shuffle(file_paths)
            for input_image, input_mask in self.take_img_as_numpy(required_postfixes=required_postfixes,
                                                                  file_paths=file_paths):
                assert input_mask is not None, "Маска не найдена"
                input_image = input_image / 255.0
                # for img, msk in self.__kernel_segmentation(input_image, input_mask.reshape(input_mask.shape + (1,))):
                for img, msk in self.__kernel_segmentation(input_image,
                                                           split_mask_values_by_channel(input_mask, self.__N_CLASSES)):
                    yield img, msk

    def __kernel_segmentation(self, image_array: np.ndarray,
                              mask_array: np.ndarray,
                              kernel_size: Optional[Tuple[int, int, int]] = None,
                              step: Optional[Tuple[int, int, int]] = None,
                              batch_size: Optional[int] = None,
                              ):
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
        if batch_size is None:
            batch_size = self.__batch_size
        if len(kernel_size) == 2:
            kernel_n = (batch_size,) + kernel_size
            step_n = (int(round(self.__SHAPE[-1] / batch_size)),) + step
        if len(kernel_size) == 3:
            kernel_n = kernel_size
            step_n = step
        img = roll_3d_over_4d_array(image_array, kernel_n, step_n)
        msk = roll_3d_over_4d_array(mask_array, kernel_n, step_n)
        i_ind, j_ind, k_ind, _, _, _, _ = img.shape
        for i in range(i_ind):
            for j in range(j_ind):
                for k in range(k_ind):
                    if len(kernel_size) == 2:
                        yield img[i, j, k, :, :, :, :], msk[i, j, k, :, :, :, :]
                    else:
                        try:
                            tmp_list_img
                        except NameError:
                            tmp_list_img = []
                        try:
                            tmp_list_msk
                        except NameError:
                            tmp_list_msk = []
                        if len(tmp_list_img) >= batch_size:
                            result_img = np.asarray(tmp_list_img, dtype=np.float32).reshape(
                                (batch_size,) + kernel_size + (img.shape[-1],))
                            result_msk = np.asarray(tmp_list_msk, dtype=np.float32).reshape(
                                (batch_size,) + kernel_size + (msk.shape[-1],))
                            tmp_list_img = []
                            tmp_list_msk = []
                            del tmp_list_img
                            del tmp_list_msk
                            gc.collect()
                            yield result_img, result_msk
                        else:
                            tmp_list_img.append(img[i, j, k, :, :, :, :])
                            tmp_list_msk.append(msk[i, j, k, :, :, :, :])


def split_mask_values_by_channel(y_true, n_channel):
    new_y_true = np.zeros(y_true.shape + (n_channel,), dtype=np.float32)

    for i in range(n_channel):
        new_y_true[..., i] = np.where(np.logical_and(y_true <= i, y_true > i - 1), 1, 0)
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
    # from pympler import asizeof
    SIZE_IMAGE = 64
    STEP_WINDOW = 64
    BATCH_SIZE = 1
    loader = PreprocessLoadData(kernel_size=(SIZE_IMAGE, SIZE_IMAGE, SIZE_IMAGE,),
                                step=(STEP_WINDOW, STEP_WINDOW, STEP_WINDOW), batch_size=BATCH_SIZE)

    loader.find_files()
    a = loader.file_paths
    test_data = loader.get_kernel_generator_data("train")
    #
    test_ind = 0
    for _ in test_data:
        print(f"{test_ind+1}/{len(a)}")
        test_ind += 1
        break

