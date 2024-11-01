import pandas as pd
import matplotlib.pyplot as plt

files = [
    'validation_accuracy_overtime_TL1000.csv',
    'validation_accuracy_overtime_TL1000_LR10x.csv',
    'validation_accuracy_overtime_TL1000_LR100x.csv',
    'validation_accuracy_overtime_TL1000_LR1000x.csv',
    'validation_accuracy_overtime_TL1000_ExtraLayer.csv'
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
    plt.plot(df['epoch'].values, df['accuracy_token'].values, label=label)

plt.xlabel('Epoch', fontsize=axis_label_fontsize)
plt.ylabel('Token Accuracy', fontsize=axis_label_fontsize)
plt.legend(loc='best', fontsize=legend_fontsize)
plt.grid(True)
plt.xticks([100, 300, 500, 700, 1000], fontsize=ticks_fontsize)
plt.yticks(fontsize=ticks_fontsize)
plt.title('Token Accuracy for PlantCyc Validation Set over Epochs', fontsize=title_fontsize)
plt.xlim(0, 1000)
plt.ylim(0.92, 1)

plt.savefig('validation_accuracy_token_biochem_1000.png')
plt.show()

plt.figure(figsize=(10, 6))
for file, label in zip(files, custom_labels):
    df = pd.read_csv(file)
    plt.plot(df['epoch'].values, df['accuracy_arc'].values, label=label)

plt.xlabel('Epoch', fontsize=axis_label_fontsize)
plt.ylabel('ARC Accuracy', fontsize=axis_label_fontsize)
plt.legend(loc='best', fontsize=legend_fontsize)
plt.grid(True)
plt.xticks([100, 300, 500, 700, 1000], fontsize=ticks_fontsize)
plt.yticks(fontsize=ticks_fontsize)
plt.title('ARC Accuracy for PlantCyc Validation Set over Epochs', fontsize=title_fontsize)
plt.xlim(0, 1000)
plt.ylim(0.88, 0.95)

plt.savefig('validation_accuracy_arc_biochem_1000.png')
plt.show()

plt.figure(figsize=(10, 6))
for file, label in zip(files, custom_labels):
    df = pd.read_csv(file)
    plt.plot(df['epoch'].values, df['accuracy_brc'].values, label=label)

plt.xlabel('Epoch', fontsize=axis_label_fontsize)
plt.ylabel('BRC Accuracy', fontsize=axis_label_fontsize)
plt.legend(loc='best', fontsize=legend_fontsize)
plt.grid(True)
plt.xticks([100, 300, 500, 700, 1000], fontsize=ticks_fontsize)
plt.yticks(fontsize=ticks_fontsize)
plt.title('BRC Accuracy for PlantCyc Validation Set over Epochs', fontsize=title_fontsize)
plt.xlim(0, 1000)
plt.ylim(0.74, 0.90)

plt.savefig('validation_accuracy_brc_biochem_1000.png')
plt.show()

print("All accuracy graphs have been saved as PNG files.")
