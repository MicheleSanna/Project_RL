from matplotlib import pyplot as plt
import numpy as np

def moving_average(array, window_size):
    return np.convolve(array, np.ones(window_size)/window_size, mode='valid')

def plot(data, smoothing_window):
    plt.plot(data, label='Average(100) episode reward', color='lightskyblue')
    plt.plot(moving_average(data, smoothing_window), label='Smoothed', color='blue')
    plt.xlabel('Episode')
    plt.ylabel('Reward')
    plt.title('Episode Reward Over Time')
    plt.show()            
