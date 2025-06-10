import numpy as np
from datasets import DatasetDict, load_from_disk
from pathlib import Path

def one_hot_labels(labels:np.ndarray) -> np.ndarray:
    """
    convert a 1D-array of labels to a 2D-array, where each column vector represents a one-hot encoded label.
    i.e, [1,0,9,8] --> [[0,1,0,0],
                        [1,0,0,0],
                        [0,0,0,0],
                        [0,0,0,0],
                        [0,0,0,0],
                        [0,0,0,0],
                        [0,0,0,0],
                        [0,0,0,0],
                        [0,0,0,1],
                        [0,0,1,0],
                        ]
    
    Args:
        labels (np.ndarray): A 1D array of integer labels (e.g., [1, 0, 9, 8]).

    Returns:
        np.ndarray: A 2D array where each column is a one-hot encoded label.
    """
    # Determine the number of unique classes (assuming labels are 0-9 for MNIST)
    num_classes = 10  # MNIST has 10 classes (digits 0-9)
    
    # Initialize a zero matrix of shape (num_classes, num_labels)
    num_labels = labels.shape[0]
    one_hot_matrix = np.zeros((num_classes, num_labels))
    
    # Fill the one-hot matrix
    for i, label in enumerate(labels):
        one_hot_matrix[label, i] = 1
    
    return one_hot_matrix

def load_samples_by_indices(dataset:DatasetDict, indices:list, subset='train', one_hot=True) -> dict:
    """
    Load samples by given indices.

    Args:
        dataset (DatasetDict):  The dataset containing the MNIST data.
        indices (int):          The indices to load the samples.
        subset(str):            The subset of the dataset, default to 'train'
        one_hot(bool):          The type of returning labels, default to one-hot encoded labels.
        

    Returns:
        {   
            flatten_images(np.ndarray):  A matrix containing the loaded samples.
            labels(np.ndarray):     A 1D-array of labels or a 2D-array of one-hot encoded samples
            one-hot(bool):          True for one-hot encoded labels, False for literal labels  
        }   
    """

    # Ensure the subset is either 'train' or 'test'
    if not (subset == 'train' or subset == 'test'):
        raise ValueError(f"Subset <{subset}> doesn't exist.")
    
    subset_size = dataset[subset].num_rows

    # Catch the IndexError and throw it again
    try:
        selected = dataset[subset].select(indices)
        
        # Extract the images and labels from the dataset
        images = selected['image']  # a list of 2D arrays of 28x28 elements
        labels = np.array(selected['label'])  # a 1d-array of labels(0-9)

        # Convert the list of images to a NumPy array
        col_vectors = np.array([np.array(img).flatten().reshape(-1,1) for img in images])
        
        # print(f'col_vectors.shape()={col_vectors.shape}')
        
        # Stack column vectors to a matrix
        matrix = np.hstack(col_vectors)
        
        # print(f'matrix.shape()={matrix.shape}')

        # Convert labels
        if one_hot:
            labels = one_hot_labels(labels)


        return {
            'flatten_images': matrix, 
            'labels': labels,
            'one_hot': one_hot
            }

    except IndexError as err: 
        raise IndexError(f"IndexError caught and thrown from function 'load_sample_by_indices()', {err}")


def load_samples_random(dataset: DatasetDict, num_samples:int, subset='train', offset=0, one_hot=True) -> dict:
    """
    Load samples from a Hugging Face DatasetDict into a NumPy matrix.

    Args:
        dataset (DatasetDict):  The dataset containing the MNIST data.
        num_samples (int):      The number of samples to load.
        subset(str):            The subset of the dataset, default to 'train'
        offset (int):           The starting index from which to load samples, default to 0. 
                                Randomly pick num_samples of samples if offset < 0
        one_hot(bool):          The type of returning labels, default to one-hot encoded labels.
        

    Returns:
        {   
            flatten_images(np.ndarray):  A matrix containing the loaded samples.
            labels(np.ndarray):     A 1D-array of labels or a 2D-array of one-hot encoded samples
            one-hot(bool):          True for one-hot encoded labels, False for literal labels  
        }   
    """

    # Ensure the subset is either 'train' or 'test'
    if not (subset == 'train' or subset == 'test'):
        raise ValueError(f"Subset <{subset}> doesn't exist.")
    
    subset_size = dataset[subset].num_rows

    if offset < 0:
        if num_samples <= 0 or num_samples >= subset_size:
            raise ValueError("Number of samples is invalid or exceeds dataset bounds.")
        indices = np.random.randint(0, subset_size, size=num_samples).tolist()
    else:
        if offset >= len(dataset[subset]):
            raise ValueError("Offset is out of bounds.")
        if num_samples <= 0 or offset + num_samples > len(dataset[subset]):
            raise ValueError("Number of samples is invalid or exceeds dataset bounds.")
        indices = [i for i in range(offset, offset+num_samples)]

    selected = dataset[subset].select(indices) 
    
    # Extract the images and labels from the dataset
    images = selected['image']  # a list of 2D arrays of 28x28 elements
    labels = np.array(selected['label'])  # a 1d-array of labels(0-9)

    # print(f'image type:{type(images[0])}\nlabel type:{type(labels[0])}')

    # Convert the list of images to a NumPy array
    col_vectors = np.array([np.array(img).flatten().reshape(-1,1) for img in images])
    # print(f'col_vectors.shape:{col_vectors.shape}')

    # Stack column vectors to a matrix
    matrix = np.hstack(col_vectors)
    # print(f'matrix.shape:{matrix.shape}')

    # Convert labels
    if one_hot:
        labels = one_hot_labels(labels)


    return {
        'flatten_images': matrix, 
        'labels': labels,
        'one_hot': one_hot
        }


def load_local(saved_path=None):
    '''
    load samples from disk, use script_dir.parent.parent / 'data/saved/mnist' as default path 
    '''

    if not saved_path:
        # Load from the default dir 
        script_dir = Path(__file__).parent
        print(script_dir)
        saved_path = script_dir.parent / "data/saved/mnist"

    if not Path.exists(saved_path) and not saved_path.is_dir():
        raise ValueError(f"Given path: {saved_path} doesn't exist.")
    
    dataset = load_from_disk(saved_path)

    return dataset



def test_load():   
    dataset = load_local()
    print("Reloaded. Train size:", len(dataset["train"]))
    print("Reloaded. Test size:", len(dataset["test"]))

    data = load_samples_random(dataset, 200, subset='train', offset=-1, one_hot=True)
    print(data.keys())
    print(data['flatten_images'].shape)
    print(data['labels'].shape)

    return

if __name__ == '__main__':
    test_load()