import random

from PIL import ImageDraw, ImageFont, Image
from data.managers import BananaDataManager
from torchvision.transforms import ToPILImage
from utils.Params import Params


def show_examples(ds, split: str, examples_per_class: int = 3, size=(224, 224)):
    tensor2pil = ToPILImage()
    w, h = size
    labels = [(0, "0"), (1, "1"), (2, "2"), (3, "3")]
    ds = list(ds)
    grid = Image.new('RGB', size=(examples_per_class * w, len(labels) * h))
    draw = ImageDraw.Draw(grid)
    font = ImageFont.truetype("arial.ttf", 26, encoding="unic")

    # AutoAugmentPolicy.IMAGENET
    # AutoAugmentPolicy.CIFAR10
    # AutoAugmentPolicy.SVHN

    for label_id, label in enumerate(labels):
        # Filter the dataset by a single label and grab a few samples
        ds_slice = random.choices([example for example in ds if example[1] == label_id], k=examples_per_class)
        # augmenter = AutoAugment(AutoAugmentPolicy.IMAGENET)
        # Plot this label's examples along a row
        for i, example in enumerate(ds_slice):
            # image = augmenter(tensor2pil(example[0]))
            image = tensor2pil(example[0])
            idx = examples_per_class * label_id + i
            box = (idx % examples_per_class * w, idx // examples_per_class * h)
            grid.paste(image.resize(size), box=box)
            # bbox = draw.textbbox(position, text, font=font)
            # draw.rectangle(box, fill="red")
            # draw.text(box, str(label), fill=(0, 0, 0, 255), font=font)
            text = f"grado: {label[1]} ({split})"
            bbox = draw.textbbox(box, text, font=font)
            draw.rectangle(bbox, fill="black")
            draw.text(box, text, font=font, fill="white")

    return grid


def main():
    _, data_params, _, _ = Params.load()
    data_manager = BananaDataManager(data_params)
    data_manager.generate_split()

    data = data_manager.load_split(0)

    train_data = data["train"].dataset
    test_data = data["test"].dataset
    val_data = data["val"].dataset

    grid = show_examples(train_data, "train", examples_per_class=3)
    # grid.save("svhn.png")
    grid.show()
    # grid = show_examples(test_data, "test", examples_per_class=3)
    # grid.show()
    # #
    # grid = show_examples(val_data, "val", examples_per_class=3)
    # grid.show()


if __name__ == "__main__":
    main()
