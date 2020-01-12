import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import image as mp_image

image_file = "data/voc/automobile/000142.jpg"
plt.close()
image = mp_image.imread(image_file)
plt.imshow(image)
plt.show()
plt.clf()

# plot a histogram
plt.hist(image.ravel())
plt.show()
plt.clf()

# checking the cumulative
plt.hist(image.ravel(), bins=255, cumulative=True)
plt.show()
plt.clf()

# contrast streching and histogram equalization
