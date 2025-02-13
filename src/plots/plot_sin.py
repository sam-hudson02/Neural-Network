import matplotlib.animation as animation
import matplotlib.pyplot as plt
from utils.alpha import exp_alpha, step_alpha
from utils.data import load_cifar_10_data, load_sin, load_mnist_data, load_cifar_data
from utils.math_data import load_math_data, load_math_meta
import numpy as np
from utils.utils import Activation, mse, mse_prime, one_hot
from models.classify import Classifier
from layers.dense import Dense
from layers.activation import Activation as ActivationLayer
from layers.convolution import Convolution
from layers.reshape import Reshape
from models.nn import Network
from utils.activation import ReLU, Sigmoid, Tanh
import matplotlib
from utils.reddit_scrape import load_reddit_data


def sin_gif(epoch: int = 500):
    x, y = load_sin()
    layers = [
        Dense(1, 128),
        ActivationLayer(ReLU()),
        Dense(128, 256),
        ActivationLayer(ReLU()),
        Dense(256, 128),
        ActivationLayer(ReLU()),
        Dense(128, 1),
    ]

    def alpha(i):
        if i < 100:
            return 0.00005
        else:
            return 0.00002

    network = Network(layers, softmax=False, loss=mse,
                      loss_prime=mse_prime, verbose=True)

    #loss_history = network.train(x, y, epoch, alpha=alpha,
                                 #batch_size=200)
    

    # predictions = network.prop(x.T)
    # print(f'Predictions: {predictions}')
    # print(f'Accuracy: {mse(y, predictions)}')
    # shift x by 2 pi

   
   
    fig, ax = plt.subplots()
    ax.set_xlim([0, 2*np.pi])
    ax.set_ylim([-1, 1])


    preds = []
    loss = []
    
    for i in range(epoch+1):
        if i % 100 == 0:
            loss_history = network.train(x, y, i, alpha=alpha,
                                 batch_size=200)
            predictions = network.prop(x.T)
            loss.append(plt.plot(np.log(loss_history), color='r'))
            preds.append(predictions.T)
            
            #plt.plot(x, y)
            #plt.plot(x, predictions.T)

    frames = len(preds)
 
    def animate(frames):
        
        plt.cla()
        ax.set_xlim([0, 2*np.pi])
        ax.set_ylim([-1-0.5, 1+0.5])
        ax.plot(x,y)
        yp = preds[frames]
        ax.plot(x,yp)
        
        
    
    
    #plt.plot(x, y, color='b')
    anim = animation.FuncAnimation(fig, func=animate, interval=500, frames=range(0, frames))
    #ani = animation.ArtistAnimation(fig, preds, interval=500,
    #                              repeat_delay=1000)
    
    plt.show()
    


    
    # plt.gca()
    # ani_2 = animation.ArtistAnimation(fig, loss, interval=500, repeat_delay=1000)
    # plt.show()
        




    
    
    

