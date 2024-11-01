import pandas as pd
import matplotlib.pyplot as plt

files = [
    'log_training_file_100.csv',
    'log_training_file_300.csv',
    'log_training_file_500.csv',
    'log_training_file_700.csv',
    'log_training_file_1000.csv'
]

custom_labels = [
    '100Epochs',
    '300Epochs',
    '500Epochs',
    '700Epochs',
    '1000Epochs'
]

axis_label_fontsize = 14
ticks_fontsize = 12
legend_fontsize = 14
title_fontsize = 18

plt.figure(figsize=(10, 6))

for file, label in zip(files, custom_labels):
    df = pd.read_csv(file)
    plt.plot(df['Epoch'].values, df['Loss'].values, label=label)

plt.xlabel('Epoch', fontsize=axis_label_fontsize)
plt.ylabel('Loss', fontsize=axis_label_fontsize)
plt.title('Loss over Epochs', fontsize=title_fontsize)
plt.legend(loc='best', fontsize=legend_fontsize)
plt.grid(True)
plt.xticks(fontsize=ticks_fontsize)
plt.yticks(fontsize=ticks_fontsize)

plt.xticks([100, 300, 500, 700, 1000])

plt.savefig('loss_epochs_behaviour.png')

plt.show()