from src.simple_network import SimpleNet
from src.load_samples import load_samples_random as load_samples, load_local



def test():

    # load samples, column vectors view. 
    # images:784x200, labels:10x200
    dataset = load_local()
    samples = load_samples(dataset, 200, subset='train', offset=-1, one_hot=True)
    images = samples['flatten_images']
    images = (images - images.mean()) / images.std()  # Zero mean, unit variance
    labels = samples['labels']
    one_hot = samples['one_hot']

    print(images.shape)
    print(labels.shape)
    print(one_hot)

    # initialize layers
    neurons = {'i':784, 'h1':16, 'h2': 16, 'o':10}

    simple_net = SimpleNet(neurons)


    print(simple_net.weights)
    print(simple_net.biases)

    acc = simple_net.accuracy(images, labels)
    print(acc)

    loss = simple_net.loss(images, labels)
    print(loss)

    grads = simple_net.gradients(images, labels)
    print(grads)

    # eta = 0.01  # learning rate

    # for name in simple_net.weights.keys():
    #     print(simple_net.weights[name] - grads['weights'][name] )
    #     print(simple_net.biases[name] - grads['biases'][name] )

if __name__ == '__main__':

    test()
