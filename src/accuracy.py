import numpy as np

def accuracy(prediction:np.ndarray, t:np.ndarray) -> float:
    '''
    calculate the accuracy between the prediction and the supervise labels
    each column vector in the matrices represent a one-hot label 

    prediction(np.ndarray): predict result matrix
    t(np.ndarray): supervise label matrix

    returns:
    acc(float): accuracy rate in decimal

    '''

    if prediction.shape != t.shape:
        raise ValueError(f"prediction's shape {prediction.shape} mismatches t's shape {t.shape}")
    
    predict_labels = np.argmax(prediction, axis=0)
    labels = np.argmax(t, axis=0)

    # print(predict_labels)
    # print(f"predict_labels' shape: {predict_labels.shape}\nlabels' shape: {labels.shape}")

    return np.sum(predict_labels == labels) / t.shape[1]

def test():

    predict = np.random.rand(10,200)
    t = np.random.rand(10,200)

    print(accuracy(predict, t))

if __name__ == '__main__' :
    test()