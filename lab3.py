# -*- coding: utf-8 -*-
"""
Created on Mon Oct 28 21:06:31 2024

@author: Owl
"""
import cv2
import numpy as np
from skimage import restoration
import matplotlib.pyplot as plt

image_path = 'cow.jpg'
image = cv2.imread(image_path)

if image is None:
    print("Ошибка: Изображение не найдено. Проверьте путь к файлу.")
else:
    h, w = image.shape[:2]

    # Указанные координаты углов изображения
    src_points = np.float32([[126, h], [w, h], [w-126, 0], [0, 0]])
    dst_points = np.float32([[0, h], [w, h], [w, 0], [0, 0]])

    # Преобразование перспективы
    M = cv2.getPerspectiveTransform(src_points, dst_points)
    aligned_image = cv2.warpPerspective(image, M, (w, h))

    # Удаление шума
    denoised_image = cv2.medianBlur(aligned_image, 5)

    # Дополнительное удаление шума волновым преобразованием
    #smoothed_image = restoration.denoise_wavelet(denoised_image, multichannel=True)
    #smoothed_image = (smoothed_image * 255).astype(np.uint8)


    brightness_enhanced = cv2.convertScaleAbs(denoised_image, alpha=1.5, beta=15)


    plt.figure(figsize=(10, 10))
    plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    plt.title('Оригинальное изображение')
    plt.axis('off')
    plt.show()

    plt.figure(figsize=(10, 10))
    plt.imshow(cv2.cvtColor(denoised_image, cv2.COLOR_BGR2RGB))
    plt.title('Сглаженное изображение')
    plt.axis('off')
    plt.show()

    plt.figure(figsize=(10, 10))
    plt.imshow(cv2.cvtColor(brightness_enhanced, cv2.COLOR_BGR2RGB))
    plt.title('Увеличенная яркость')
    plt.axis('off')
    plt.show()








