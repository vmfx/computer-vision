import cv2
import numpy as np

image = cv2.imread("name.jpg")

height, width = image.shape[:2]

# 1. Зсув зображення на 10 пікселів вправо та 20 вниз
shift_filter = np.float32([[1, 0, 10], [0, 1, 20]])
shifted_image = cv2.warpAffine(image, shift_filter, (width, height))

# 2. Інверсія
inverted_image = cv2.bitwise_not(image)

# 3. Згладжування по Гауссу
gaussian_filter = np.array([[1, 2, 1], [2, 4, 2], [1, 2, 1]], dtype=np.float32) / 16
gaussian_image = cv2.filter2D(image, -1, gaussian_filter)

# 4. Розмиття "рух по діагоналі"
diagonal_blur_filter = np.array([[1, 1, 1], [1, 1, 1], [1, 1, 1]], dtype=np.float32) / 9
diagonal_blurred_image = cv2.filter2D(image, -1, diagonal_blur_filter)

# 5. Підвищення різкості
sharpen_filter = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]], dtype=np.float32)
sharpened_image = cv2.filter2D(image, -1, sharpen_filter)

# 6. Фільтр Собеля (вертикальний)
sobel_filter = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=np.float32)
sobel_image = cv2.filter2D(image, -1, sobel_filter)

# 7. Фільтр границь
edge_filter = np.array([[-1, -1, -1], [-1, 8, -1], [-1, -1, -1]], dtype=np.float32)
edged_image = cv2.filter2D(image, -1, edge_filter)

# 8. Мій фільтр
def emboss_filter(image):
    kernel_emboss = np.array([[-2, -1, 0], [-1, 1, 1], [0, 1, 2]])
    return cv2.filter2D(image, -1, kernel_emboss)

embossed_image = emboss_filter(image)

# Відображення та збереження результатів
cv2.imshow("Original Image", image)
cv2.imshow("Shifted Image", shifted_image)
cv2.imshow("Inverted Image", inverted_image)
cv2.imshow("Gaussian Image", gaussian_image)
cv2.imshow("Diagonal Blurred Image", diagonal_blurred_image)
cv2.imshow("Sharpened Image", sharpened_image)
cv2.imshow("Sobel Image", sobel_image)
cv2.imshow("Edged Image", edged_image)
cv2.imshow("Embossed Image", embossed_image)

cv2.waitKey(0)
cv2.destroyAllWindows()

