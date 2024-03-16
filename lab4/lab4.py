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


# Box average 
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
                filtered_image[i, j] = image[i, j]  # Якщо сума значень оточення нуль, залишаємо піксель без змін

    return filtered_image



# Зчитуємо зображення
img = cv2.imread("name.jpg")

# Додавання імпульсного шуму
salt_pepper_noisy_image = add_salt_pepper_noise(img, salt_prob=0.01, pepper_prob=0.01)

# Відображення зображення з імпульсним шумом
cv2.imshow('Salt-Pepper Noisy Image', salt_pepper_noisy_image)
cv2.waitKey(0)
cv2.destroyAllWindows()

# Додавання гаусового шуму
gaussian_noisy_image = add_gaussian_noise(img, mean=0, std_dev=25)

# Відображення зображення з гаусовим шумом
cv2.imshow('Gaussian Noisy Image', gaussian_noisy_image)
cv2.waitKey(0)
cv2.destroyAllWindows()

# Фільтрація зображень за допомогою OpenCV після додавання шуму

# Фільтр Медіани
median_filtered_image = median_filter(salt_pepper_noisy_image, kernel_size=3)

# Відображення зображення після фільтрації Медіаною
cv2.imshow('Median Filtered Image', median_filtered_image)
cv2.waitKey(0)
cv2.destroyAllWindows()

# Фільтр Зваженої медіани
weighted_median_filtered_image = weighted_median_filter(salt_pepper_noisy_image, kernel_size=3)

# Відображення зображення після фільтрації Зваженою медіаною
cv2.imshow('Weighted Median Filtered Image', weighted_median_filtered_image)
cv2.waitKey(0)
cv2.destroyAllWindows()

# Фільтр Згладжування коробкою
box_filtered_image = box_average_filter(gaussian_noisy_image, kernel_size=5)

# Відображення зображення після фільтрації Згладжуванням коробкою
cv2.imshow('Box Filtered Image', box_filtered_image)
cv2.waitKey(0)
cv2.destroyAllWindows()


