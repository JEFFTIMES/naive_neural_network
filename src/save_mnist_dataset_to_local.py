
from random import randint
from pathlib import Path
from datasets import load_dataset, load_from_disk
import numpy as np
import PIL
import matplotlib.pyplot as plt

def save_to_local(path:Path=None) -> None :
    dir = path if path else Path(__file__).parent
    # Step 1: Load and cache
    cache_path = dir.parent.parent / "data/cache/mnist"

    # Ensure the cache directory exists
    cache_path.mkdir(parents=True, exist_ok=True)

    dataset = load_dataset("mnist", cache_dir=cache_path, streaming=False)
    print("Cached to:", cache_path)

    # Step 2: Save explicitly
    saved_path = dir.parent.parent / "data/saved/mnist"
    dataset.save_to_disk(saved_path)
    print("Saved to:", saved_path)


def display_samples(path:Path=None) -> None:
    dir = path if path else Path(__file__).parent
    saved_path = dir.parent.parent / "data/saved/mnist"

    # Load the MNIST dataset
    dataset = load_from_disk(saved_path)

    len_train_set = len(dataset['train'])
    len_test_set = len(dataset['test'])

    print(f'Reloaded. Train set size: {len_train_set}, Test set size: {len_test_set}')
    
    train_sample_idx = randint(0,len_train_set-1)
    test_sample_idx = randint(0, len_test_set-1)
    
    # samples 
    samples = []
    samples.append((dataset['train'][train_sample_idx]['image'], dataset['train'][train_sample_idx]['label']))
    samples.append((dataset['test'][test_sample_idx]['image'], dataset['test'][test_sample_idx]['label']))

   
    for sample in samples:
        img_array = np.array(sample[0])
        print(img_array.shape)

        # plot the image
        plt.imshow(img_array, cmap='gray')  # Use 'gray' colormap for grayscale
        plt.title(f"Image: {sample[1]}")
        plt.show()

if __name__ == '__main__':

    save_to_local()
    display_samples()

