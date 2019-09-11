import os
import matplotlib.pyplot as plt
import numpy as np
import shutil
from tqdm import tqdm
from tensorboardX import SummaryWriter
import time

def visual_results_plt(file_path):
    file_list = os.listdir(file_path)
    file_list = [file for file in file_list if file.endswith('.npy')]
    model = []
    loss = []
    regression_loss = []
    classification_loss = []
    correct = []
    wrong = []
    miss = []
    map = []
    for file in file_list:
        data = np.load(f'{file_path}/{file}')
        data = data.tolist()
        # for da in data:
        #     print(da)
        print(f'model : {data["model"]}-loss : {data["loss"]}-reg loss : {data["regression_loss"]}-'
              f'cla loss : {data["classification_loss"]}-correct : {data["correct_"]/data["count_"]}-'
              f'wrong : {data["wrong_"]/data["count_"]}-miss : {data["miss_"]/data["count_"]}-'
              f'map : {data["mAP"]}')

        model.append(data['model'])
        loss.append(data['loss'])
        regression_loss.append(data['regression_loss'])
        classification_loss.append(data['classification_loss'])
        correct.append(data['correct_']/data['count_'])
        wrong.append(data['wrong_']/data['count_'])
        miss.append(data['miss_']/data['count_'])
        map.append(data['mAP'])

    fig = plt.figure(figsize=(16,10))
    ax1 = fig.add_subplot(3, 1, 1)
    ax2 = fig.add_subplot(3, 1, 2)
    ax3 = fig.add_subplot(3, 1, 3)
    plt.xticks(rotation=90)
    ax1.plot(model, loss, c='red', label='loss')
    ax1.plot(model, regression_loss, c='green', label='regression_loss')
    ax1.plot(model, classification_loss, c='blue', label='classification_loss')
    ax2.plot(model, correct, c='green', label='correct')
    ax2.plot(model, wrong, c='blue', label='wrong')
    ax2.plot(model, miss, c='red', label='miss')
    ax3.plot(model, map, c='black', label='mAP')

    plt.show()

def visual_results_board(file_path):
    current_time = time.strftime('%y%m%d%H%M')
    writer = SummaryWriter(f'{file_path}/{current_time}')
    file_list = [file for file in os.listdir(file_path) if file.endswith('.npy') ]
    print(f'File number : {len(file_list)}')
    for file in file_list:
        if int(file[:3])%10 != 0: continue
        print(file)
        data = np.load(f'{file_path}/{file}')
        data = data.tolist()
        model = data["model"][:-4]
        loss = round(float(data["loss"]), 4)
        regression_loss = round(float(data["regression_loss"]), 4)
        classification_loss = round(float(data["classification_loss"]), 4)
        correct = round(data["correct_"] / data["count_"], 4)
        wrong = round(data["wrong_"] / data["count_"], 4)
        miss = round(data["miss_"] / data["count_"], 4)
        map = round(data["mAP"], 4)

        writer.add_scalars('Detection_Data/LOSS',
                           {'mAP': map, 'loss': loss,
                            'regression_loss': regression_loss,
                            'classification_loss': classification_loss,
                            'Correct detection rate': correct,
                            'Detection rate': (1 - miss)}, model)
        name = []
        value = []
        for d in data:
            if d.endswith('-ap'):
                name.append(d[:-3])
                value.append(round(data[d], 4))
        writer.add_scalars('Detection_Data/AP', {class_name:ap for class_name, ap in zip(name, value)}, model)
    writer.close()


if __name__ == '__main__':
    file_path = r''
    visual_results_board(file_path)