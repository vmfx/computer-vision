import cv2
import numpy as np

# Імпульсний шум
def add_salt_pepper_noise(image, salt_prob, pepper_prob):
    noisy_image = np.copy(image)
    salt_mask = np.random.random(image.shape[:2]) < salt_prob
    pepper_mask = np.random.random(image.shape[:2]) < pepper_prob
    noisy_image[salt_mask] = 255
    noisy_image[pepper_mask] = 0
    return noisy_image


# Нормальний шум
def add_gaussian_noise(image, mean, std_dev):
    noisy_image = np.copy(image)
    h, w, c = image.shape
    noise = np.random.normal(mean, std_dev, (h, w, c))
    noisy_image = np.clip(noisy_image.astype(np.float64) + noise, 0, 255).astype(np.uint8)
    return noisy_image


# Згладжування коробкою
def box_average_filter(image, kernel_size):
    return cv2.blur(image, (kernel_size, kernel_size))


# Медіана
def median_filter(image, kernel_size):
    return cv2.medianBlur(image, kernel_size)


# Зважена медіана
def weighted_median_filter(image, kernel_size):
    padded_image = cv2.copyMakeBorder(image, kernel_size // 2, kernel_size // 2, kernel_size // 2, kernel_size // 2,
                                      cv2.BORDER_CONSTANT, value=(0, 0, 0))
    filtered_image = np.zeros_like(image)
    h, w, _ = image.shape

    for i in range(h):
        for j in range(w):
            neighborhood = padded_image[i:i + kernel_size, j:j + kernel_size].reshape(-1, 3)
            neighborhood_values = neighborhood[:, 0]
            sum_neighborhood_values = neighborhood_values.sum()
            if sum_neighborhood_values != 0:
                weights = neighborhood_values / sum_neighborhood_values
                sorted_indices = np.argsort(neighborhood_values)
                cumulative_weights = np.cumsum(weights[sorted_indices])
                median_index = np.argmax(cumulative_weights >= 0.5)
                filtered_image[i, j] = neighborhood[sorted_indices[median_index]]
            else:
                filtered_image[i, j] = image[i, j]  

    return filtered_image


# Функція фільтрації у HSV представлені
def filter_hsv(image):
    hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    # Тут можна додати будь-які фільтри для каналів HSV
    return hsv_image


img = cv2.imread("name.jpg")

# Додавання імпульсного шуму
salt_pepper_noisy_image = add_salt_pepper_noise(img, salt_prob=0.01, pepper_prob=0.01)

# Додавання нормального шуму
gaussian_noisy_image = add_gaussian_noise(img, mean=0, std_dev=25)


# Фільтр Медіани
median_filtered_image_hsv = median_filter(filter_hsv(salt_pepper_noisy_image), kernel_size=3)

# Фільтр Зваженої медіани
weighted_median_filtered_image_hsv = weighted_median_filter(filter_hsv(salt_pepper_noisy_image), kernel_size=3)

# Фільтр Згладжування коробкою
box_filtered_image_hsv = box_average_filter(filter_hsv(gaussian_noisy_image), kernel_size=5)

salt_pepper_filtered_image_hsv = filter_hsv(salt_pepper_noisy_image)
gaussian_filtered_image_hsv = filter_hsv(gaussian_noisy_image)

# Відображення зображень
cv2.imshow('Salt-Pepper Filtered Image (HSV)', salt_pepper_filtered_image_hsv)
cv2.imshow('Gaussian Filtered Image (HSV)', gaussian_filtered_image_hsv)
cv2.imshow('Median Filtered Image (HSV)', median_filtered_image_hsv)
cv2.imshow('Weighted Median Filtered Image (HSV)', weighted_median_filtered_image_hsv)
cv2.imshow('Box Filtered Image (HSV)', box_filtered_image_hsv)
cv2.waitKey(0)
cv2.destroyAllWindows()
