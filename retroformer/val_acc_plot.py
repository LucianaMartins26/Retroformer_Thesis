import pandas as pd
import matplotlib.pyplot as plt

files = [
    'validation_accuracy_overtime_100.csv',
    'validation_accuracy_overtime_300.csv',
    'validation_accuracy_overtime_500.csv',
    'validation_accuracy_overtime_700.csv',
    'validation_accuracy_overtime_1000.csv'
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
    plt.plot(df['epoch'].values, df['accuracy_token'].values, label=label)

plt.xlabel('Epoch', fontsize=axis_label_fontsize)
plt.ylabel('Token Accuracy', fontsize=axis_label_fontsize)
plt.title('Token Accuracy for PlantCyc Validation Set over Epochs', fontsize=title_fontsize)
plt.legend(loc='best', fontsize=legend_fontsize)
plt.grid(True)
plt.xticks([100, 300, 500, 700, 1000], fontsize=ticks_fontsize)
plt.yticks(fontsize=ticks_fontsize)
plt.xlim(0, 1000)
plt.ylim(0.825, 0.975)

plt.savefig('validation_accuracy_token.png')
plt.show()

plt.figure(figsize=(10, 6))
for file, label in zip(files, custom_labels):
    df = pd.read_csv(file)
    plt.plot(df['epoch'].values, df['accuracy_arc'].values, label=label)

plt.xlabel('Epoch', fontsize=axis_label_fontsize)
plt.ylabel('ARC Accuracy', fontsize=axis_label_fontsize)
plt.title('ARC Accuracy for PlantCyc Validation Set over Epochs', fontsize=title_fontsize)
plt.legend(loc='best', fontsize=legend_fontsize)
plt.grid(True)
plt.xticks([100, 300, 500, 700, 1000], fontsize=ticks_fontsize)
plt.yticks(fontsize=ticks_fontsize)
plt.xlim(0, 1000)
plt.ylim(0.875, 0.925)

plt.savefig('validation_accuracy_arc.png')
plt.show()

plt.figure(figsize=(10, 6))
for file, label in zip(files, custom_labels):
    df = pd.read_csv(file)
    plt.plot(df['epoch'].values, df['accuracy_brc'].values, label=label)

plt.xlabel('Epoch', fontsize=axis_label_fontsize)
plt.ylabel('BRC Accuracy', fontsize=axis_label_fontsize)
plt.title('BRC Accuracy for PlantCyc Validation Set over Epochs', fontsize=title_fontsize)
plt.legend(loc='best', fontsize=legend_fontsize)
plt.grid(True)
plt.xticks([100, 300, 500, 700, 1000], fontsize=ticks_fontsize)
plt.yticks(fontsize=ticks_fontsize)
plt.xlim(0, 1000)
plt.ylim(0.78, 0.85)

plt.savefig('validation_accuracy_brc.png')
plt.show()

print("All accuracy graphs have been saved as PNG files.")
