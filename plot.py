from matplotlib import pyplot as plt
import numpy as np

def moving_average(array, window_size):
    return np.convolve(array, np.ones(window_size)/window_size, mode='valid')

def moving_sum(array, window_size):
    return np.convolve(array, np.ones(window_size), mode='valid')

def plot_rewards(data, smoothing_window):
    plt.plot(data, label='Average(100) episode reward', color='lightskyblue')
    plt.plot(moving_average(data, smoothing_window), label='Smoothed', color='blue')
    plt.xlabel('Episode')
    plt.ylabel('Reward')
    plt.title('Episode Reward Over Time')
    plt.legend(loc='upper right')
    plt.show()

def plot_sum(data1, data2, window_size, title, y_label):
    plt.plot(moving_sum(data1, window_size), label='Hero', color='red')
    plt.plot(moving_sum(data2, window_size), label='Opponent', color='blue')
    plt.xlabel('Episode')
    plt.ylabel(y_label)
    plt.title(title)
    plt.legend(loc='upper right')
    plt.show()

def plot_average(data1, data2, window_size, title, y_label):
    plt.plot(moving_average(data1, window_size), label='Smoothed RL player', color='red')
    plt.plot(moving_average(data2, window_size), label='Smoothed Adv player', color='blue')
    plt.xlabel('Episode')                                      
    plt.ylabel(y_label)
    plt.title(title)
    plt.legend(loc='upper right')
    plt.show()           
