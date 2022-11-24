import os.path
import time

import matplotlib as mpl
import matplotlib.pyplot as plt

from segmentation.scikit_test import rescale

mpl.rcParams['figure.dpi'] = 300
from cellpose import io, models, plot

files = ["segmentation/image.png"]
out_files = [os.path.join("segmentation/out", os.path.basename(f)) for f in files][0]

# REPLACE FILES WITH YOUR IMAGE PATHS
# files = ['img0.tif', 'img1.tif']

# view 1 image
img = io.imread(files[-1])

img = rescale(img)

plt.figure(figsize=(2, 2))
plt.imshow(img)
plt.axis('off')
plt.show()

model = models.Cellpose(gpu=True, model_type='cyto2')

# define CHANNELS to run segementation on
# grayscale=0, R=1, G=2, B=3
# channels = [cytoplasm, nucleus]
# if NUCLEUS channel does not exist, set the second channel to 0
# channels = [0,0]
# IF ALL YOUR IMAGES ARE THE SAME TYPE, you can give a list with 2 elements
# channels = [0,0] # IF YOU HAVE GRAYSCALE
# channels = [2,3] # IF YOU HAVE G=cytoplasm and B=nucleus
# channels = [2,1] # IF YOU HAVE G=cytoplasm and R=nucleus

# or if you have different types of channels in each image
channels = [0, 0]

# if diameter is set to None, the size of the cells is estimated on a per image basis
# you can set the average cell `diameter` in pixels yourself (recommended)
# diameter can be a list or a single number for all images

# you can run all in a list e.g.
imgs = [io.imread(filename) for filename in files][0]
start = time.perf_counter()
masks, flows, styles, diams = model.eval(imgs, diameter=30.0, channels=channels, batch_size=2)
end = time.perf_counter()
print(f"Time: {end - start} s")
io.masks_flows_to_seg(imgs, masks, flows, diams, out_files, channels)
io.save_to_png(imgs, masks, flows, out_files)

for i in range(len(files)):
    # img, masks, flows = imgs[i], masks[i], flows[i]
    fig = plt.figure(figsize=(12, 5))
    plot.show_segmentation(fig, img, masks, flows[0], channels=channels)
    plt.tight_layout()
    plt.show()
