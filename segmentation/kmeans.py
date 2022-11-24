import time

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from numpy import asarray
from sklearn.cluster import KMeans

im = Image.open("segmentation/image.png")
w, h = im.size

scaled_w = 512
scaled_h = h * scaled_w // w

im = im.resize((scaled_w, scaled_h), Image.Resampling.LANCZOS)
plt.imshow(im)

data = asarray(im)
s = data.shape
plt.imshow(data)
# plt.show()

k = 2
img_features = data.reshape((-1, 3))
kmeans = KMeans(n_clusters=k, max_iter=2)
start = time.perf_counter()
pred = kmeans.fit_predict(img_features)
end = time.perf_counter()
print(f"Time: {end - start} s")
# pred = kmeans.predict(X)

c = kmeans.cluster_centers_.reshape(k, 3).astype(int)
l = np.sum(c * [0.2126, 0.7152, 0.0722], axis=-1)
banana_c = np.argmax(l)
print(banana_c)

mask = np.zeros_like(img_features)
mask[pred == banana_c] = 255
# mask = np.where(pred == banana_c, np.full(3, 255), np.zeros(3))
# for x in range(X.shape[0]):
#     mask[x] = c[0, pred[x]]
mask = mask.reshape(s)

out_img = Image.fromarray(mask)
out_img = out_img.resize((w, h), Image.Resampling.LANCZOS)
print(out_img.size)

plt.imshow(out_img)
plt.show()
