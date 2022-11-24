import time

from PIL import Image
from fastseg import MobileV3Small  # MobileV3Large

# https://pypi.org/project/fastseg/

model = MobileV3Small.from_pretrained().cuda()
model.eval()

# Open a local image as input
image = Image.open('segmentation/image.png')

start = time.perf_counter()
# Predict numeric labels [0-18] for each pixel of the image
labels = model.predict_one(image)
end = time.perf_counter()
print(f"Time: {end - start} s")
