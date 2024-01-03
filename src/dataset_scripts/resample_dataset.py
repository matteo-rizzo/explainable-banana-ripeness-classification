import os
import shutil

from sklearn.utils.random import sample_without_replacement
from tqdm import tqdm


def main():
    random_seed = 0
    percentage = 80
    path_to_new_dataset = os.path.join("dataset", f"treviso-market-244_244-{percentage}")
    path_to_dataset = os.path.join("dataset", "treviso-market-224_224")

    print(f"\n Resampling dataset '{path_to_dataset}' to {percentage}%")

    for label in os.listdir(path_to_dataset):
        path_to_label = os.path.join(path_to_dataset, label)
        path_to_new_label = os.path.join(path_to_new_dataset, label)
        os.makedirs(path_to_new_label)

        num_files = len(os.listdir(path_to_label))
        sample_size = (percentage * num_files) // 100

        samples = sample_without_replacement(num_files, sample_size, random_state=random_seed)

        label_data = os.listdir(path_to_label)
        for i, img in tqdm(enumerate(samples)):
            path_to_source = os.path.join(path_to_label, label_data[i])
            path_to_destination = os.path.join(path_to_new_label, label_data[i])
            shutil.copy(path_to_source, path_to_destination)


if __name__ == "__main__":
    main()
