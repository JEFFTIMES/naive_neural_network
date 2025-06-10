from src.simple_network import SimpleNet
from src.load_samples import load_samples_random, load_samples_by_indices, load_local
from datasets import DatasetDict
from math import ceil
import matplotlib.pyplot as plt
import time
import random

def simple_network_learning(network:SimpleNet, train_data:DatasetDict, lr:int=0.1, epochs:int=120, batch_sz:int=100) -> dict:
    '''
    using stochastic gradient descent method to train the network

    receives:
    network(SimpleNet): the network to train
    train_data(DatasetDict): the training data, samples and labels
    lr(int): the learning rate, default to 0.1
    epochs(int): how many epochs to iterate, default to 24
    batch_sz(int): the mini-batch size, default to 200

    returns:
    report(dict): {
        'weights':{'name':w_matrix},...}, 
        'biases':{'name':b_matrix},...}, 
        accuracies:[(i_of_epoch,acc),...]
        }

    '''

    if not isinstance(train_data, DatasetDict):
        raise TypeError(f'given dataset is not an instance of DatasetDict.')
    
    # size of the data
    train_sz = len(train_data['train'])
    test_sz = len(train_data['test'])
    print(f'train:{train_sz}, test:{test_sz}')

    # initialize the indices
    indices = [i for i in range(train_sz)]

    # # iteration setting
    batches = int(train_sz / batch_sz if train_sz % batch_sz == 0 else train_sz // batch_sz)

    # accuracies
    accuracies = []
    max_acc = 0.0

    start = time.time()

    # iterating {epochs} round with SGD method
    for epoch in range(epochs):

        # shuffle the indices of the train set for each epoch
        random.shuffle(indices)

        for batch in range(batches):
            offset = batch * batch_sz
            # load a mini-batch of training samples and labels
            samples = load_samples_by_indices(train_data, indices[offset:offset+batch_sz], subset = 'train', one_hot = True)
            images = samples['flatten_images'] / 255.0
            # images = (images - images.mean()) / images.std()  # Zero mean, unit variance
            labels = samples['labels']
        
            # SGD
            grads = network.gradients(images, labels)
        
            # update network's weights and biases
            for name in network.weights.keys():
                network.weights[name] -= lr * grads['weights'][name]
                network.biases[name] -= lr * grads['biases'][name]
        
            
            
        t_samples = load_samples_random(train_data, batch_sz*10, subset = 'test', offset= -1, one_hot= True)
        t_images = t_samples['flatten_images'] / 255.0
        # t_images = (t_images - t_images.mean()) / t_images.std()
        t_labels = t_samples['labels']

        cur_acc = network.accuracy(t_images, t_labels)
        accuracies.append((epoch,cur_acc))
        print(f'Epoch:{epoch}, Acc:{cur_acc}')

        if cur_acc > max_acc:
            max_acc = cur_acc
            best_weights = network.weights.copy()
            best_biases = network.biases.copy()
            from_epoch = epoch
            from_batch = batch
    
    elapse = time.time() - start
    print(f'training time: {elapse} sec')


    return {
        'weights':best_weights, 
        'biases':best_biases, 
        'from_epoch': from_epoch,
        'from_batch': from_batch,
        'accuracies':accuracies
        }

def main():
    
    # load data
    data = load_local()

    # initialize network
    neurons = {'i':784, 'h1':100, 'o':10}
    network = SimpleNet(neurons)
    report = simple_network_learning(network, data, batch_sz=200)
    
    # check weights and biases
    epoch = report['from_epoch']
    batch = report['from_batch']
    best_acc = report['accuracies'][epoch]
    print(f"Best accuracy: {best_acc} comes from epoch: {epoch}, bath:{batch}")

    # print accuracies
    print(report['accuracies'])

    # plot accuracies
    x_values = [coord[0] for coord in report['accuracies']]
    y_values = [coord[1] for coord in report['accuracies']]

    # Create the plot
    plt.figure(figsize=(10, 6))  # Optional: Set figure size
    plt.plot(x_values, y_values, marker='.', linestyle='-', color='b', label='Accuracy')


    # Customize the plot
    plt.title("Accuracy as to Epoch")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.grid(True)
    plt.legend()

    # Show the plot
    plt.show()

main()