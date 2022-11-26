import matplotlib.pyplot as plt
import tools


labels, digits = tools.read_data()
img_size = 28
plt.imshow(digits[0].reshape(img_size, img_size))
plt.show()
