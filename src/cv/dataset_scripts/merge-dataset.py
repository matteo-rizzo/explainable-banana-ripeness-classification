import os
import shutil


def main():
    num_classes = 4
    path_to_raw = os.path.join("../../../dataset", "raw")
    path_to_preprocessed = os.path.join("../../../dataset", "preprocessed")
    counters = {str(i): int(0) for i in range(1, num_classes + 1)}

    paths_to_classes = {}
    for i in range(1, num_classes + 1):
        path_to_class = os.path.join(path_to_preprocessed, str(i))
        os.makedirs(path_to_class, exist_ok=True)
        paths_to_classes[str(i)] = path_to_class

    for folder in os.listdir(path_to_raw):
        path_to_acquisitions = os.path.join(path_to_raw, folder)
        for acquisition in os.listdir(path_to_acquisitions):
            path_to_images = os.path.join(path_to_acquisitions, acquisition)
            for image in os.listdir(path_to_images):
                path_to_image = os.path.join(path_to_images, image)
                label = image.split("_")[-1].split(".")[0]
                image_destination = os.path.join(paths_to_classes[label], str(counters[label]) + ".png")
                shutil.move(path_to_image, image_destination)
                counters[label] += 1


if __name__ == "__main__":
    main()
