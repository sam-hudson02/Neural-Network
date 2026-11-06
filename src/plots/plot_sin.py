import matplotlib.animation as animation
import matplotlib.pyplot as plt
from utils.data import load_sin
import numpy as np
from utils.utils import mse, mse_prime
from layers.dense import Dense
from layers.activation import Activation as ActivationLayer
from models.nn import Network
from utils.activation import ReLU


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

    network = Network(layers, softmax=False, loss=mse,
                      loss_prime=mse_prime, verbose=True)
    

    # predictions = network.prop(x.T)
    # print(f'Predictions: {predictions}')
    # print(f'Accuracy: {mse(y, predictions)}')
    # shift x by 2 pi

   
   
    fig, ax = plt.subplots()
    ax.set_xlim([0, 2*np.pi])
    ax.set_ylim([-1, 1])


    preds = []
    loss = []
    
    # each frame trains a further step epochs on the same network, so frame k
    # shows the fit after k * step epochs
    step = 100
    for i in range(0, epoch + 1, step):
        if i:
            network.train(x, y, epochs=step, batch_size=200)
            loss.append(plt.plot(np.log(network.average_loss()), color='r'))
        predictions = network.prop(x.T)
        preds.append(predictions.T)

    frames = len(preds)
 
    def animate(frames):
        
        plt.cla()
        ax.set_xlim([0, 2*np.pi])
        ax.set_ylim([-1-0.5, 1+0.5])
        ax.plot(x,y)
        yp = preds[frames]
        ax.plot(x,yp)
        
        
    
    
    #plt.plot(x, y, color='b')
    self_ref_anim = animation.FuncAnimation(fig, func=animate, interval=500, frames=range(0, frames))
    #ani = animation.ArtistAnimation(fig, preds, interval=500,
    #                              repeat_delay=1000)
    
    plt.show()
    return self_ref_anim

    # plt.gca()
    # ani_2 = animation.ArtistAnimation(fig, loss, interval=500, repeat_delay=1000)
    # plt.show()
        




    
    
    

