import random
import matplotlib.pyplot as plt

class Gantt():
    def __init__(self, total_n_job, num_mch):
        super(Gantt, self).__init__()
        self.total_n_job = total_n_job
        self.num_mch = num_mch
        self.init_plt()

    def Color(self, n):
        Color_bits = ['1', '2', '3', '4', '5', '6', '7', '8', '9', 'a', 'b', 'c', 'd', 'e', 'f']
        colours = []
        random.seed(234)
        for i in range(n):
            colour_bits = ['#']
            colour_bits.extend(random.sample(Color_bits, 6))
            colours.append(''.join(colour_bits))
        return colours

    def init_plt(self):
        plt.figure(figsize=((self.total_n_job * 1.5, self.num_mch)))
        y_value = list(range(1, 11))

        plt.xlabel('Makespan', size=20, fontdict={'family': 'SimSun'})
        plt.ylabel('Machine', size=20, fontdict={'family': 'SimSun'})
        plt.yticks(y_value, fontproperties='Times New Roman', size=20)
        plt.xticks(fontproperties='Times New Roman', size=20)
        plt.close()

    def gantt_plt(self, job, operation, mach_a, start_time, dur_a, num_job):
        Colors = self.Color(num_job)
        plt.barh(mach_a + 1, dur_a, 0.5, left=start_time, Color=Colors[job])