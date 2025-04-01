import cv2
import numpy as np

def compute_variance(depth, kernel_size=5):
    mean = cv2.blur(depth, (kernel_size, kernel_size))
    mean_sq = cv2.blur(depth ** 2, (kernel_size, kernel_size))
    variance = mean_sq - mean ** 2
    return variance

# Load depth image (grayscale where intensity represents depth)
depth = cv2.imread("depth_map.png", cv2.IMREAD_GRAYSCALE)
if depth is None:
    print("Error: Could not load depth map!")
    exit()

# Compute gradients (for slope detection)
grad_x = cv2.Sobel(depth, cv2.CV_32F, 1, 0, ksize=3)
grad_y = cv2.Sobel(depth, cv2.CV_32F, 0, 1, ksize=3)
gradient = cv2.magnitude(grad_x, grad_y)
# Compute Laplacian (for pit detection)
laplacian = cv2.Laplacian(depth, cv2.CV_32F)

# Compute local variance (for rough terrain detection)
variance = compute_variance(depth, 5)

# Thresholding
obstacles = np.zeros_like(depth, dtype=np.uint8)
obstacles[gradient > 50] = 255  # Hills/steep slopes
obstacles[laplacian < -10] = 255  # Pits
obstacles[variance > 500] = 255  # Boulders

# Save and display result
cv2.imwrite("traversability_map.png", obstacles)
cv2.imshow("Obstacles", obstacles)
cv2.waitKey(0)
cv2.destroyAllWindows()