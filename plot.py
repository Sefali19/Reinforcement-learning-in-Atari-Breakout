import matplotlib.pyplot as plt
import os
from datetime import datetime

class LivePlot:
    def __init__(self):
        self.eps_data = None
        self.fig, self.ax = plt.subplots()
        self.ax.set_xlabel('Epoch x10')
        self.ax.set_ylabel('Returns')
        self.ax.set_title("Returns over Epoch")

        self.data = None

        self.epoch = 0

    def update_plot(self,stats):
        self.data = stats['AvgReturns']
        self.eps_data = stats['EpsilonCheckpoint']
        self.epoch = len(self.data)

        self.ax.clear()
        self.ax.set_xlim([0,self.epoch])
        self.ax.plot(self.data, 'b-', label='Returns')
        self.ax.plot(self.eps_data, 'r-', label='EpsilonCheckpoint')
        self.ax.legend(loc='upper left')

        #ensure directory exists
        if not os.path.exists('plots'):
            os.makedirs('plots')

        current_date = datetime.now().strftime('%Y-%m-%d')

        self.fig.savefig(f'plots/plot_{current_date}.png')



    def plot(self):
        pass
