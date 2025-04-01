import cv2
import numpy as np
import matplotlib.pyplot as plt

height, width = 256, 256
depth_map = np.ones((height, width), dtype=np.uint8) * 128  # Base depth level

for i in range(50, 150):
    depth_map[i, 50:200] = np.clip(depth_map[i, 50:200] + (i - 50) // 2, 0, 255)

cv2.circle(depth_map, (180, 180), 30, 50, -1)

cv2.rectangle(depth_map, (80, 80), (100, 100), 200, -1)
cv2.imwrite("depth_map.png", depth_map)

# Display the depth map
plt.imshow(depth_map, cmap='gray')
plt.title("Synthetic Depth Map")
plt.axis("off")
plt.show()