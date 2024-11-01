import pandas as pd
import matplotlib.pyplot as plt

files = [
    'log_training_file_TL1000.csv',
    'log_training_file_TL1000_LR10x.csv',
    'log_training_file_TL1000_LR100x.csv',
    'log_training_file_TL1000_LR1000x.csv',
    'log_training_file_TL1000_ExtraLayer.csv'
]

custom_labels = [
    'TL1000Epochs_1e-4',
    'TL1000Epochs_1e-5',
    'TL1000Epochs_1e-6',
    'TL1000Epochs_1e-7',
    'TL1000Epochs_ExtraLayer'
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

plt.savefig('loss_epochs_behaviour_biochem_1000.png')

plt.show()

print("Loss graph has been saved as a PNG file.")
