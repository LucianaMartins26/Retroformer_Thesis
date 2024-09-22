import pandas as pd
import matplotlib.pyplot as plt

files = [
    'validation_accuracy_overtime_TL700.csv',
    'validation_accuracy_overtime_TL700_LR10x.csv',
    'validation_accuracy_overtime_TL700_LR100x.csv',
    'validation_accuracy_overtime_TL700_LR1000x.csv',
    'validation_accuracy_overtime_TL700_ExtraLayer.csv'
    ]

custom_labels = [
    'TL700Epochs',
    'TL700Epochs_1e-5',
    'TL700Epochs_1e-6',
    'TL700Epochs_1e-7',
    'TL700Epochs_ExtraLayer'
]

plt.figure(figsize=(10, 6))

for file, label in zip(files, custom_labels):
    df = pd.read_csv(file)
    plt.plot(df['epoch'], df['accuracy_token'], label=label)

plt.xlabel('Epoch')
plt.ylabel('Accuracy Token')
plt.title('Validation Accuracy Token over Epochs')
plt.legend(loc='best')
plt.grid(True)
plt.xticks([100, 300, 500, 700, 700])
plt.xlim(0, 700)
plt.ylim(0.92, 1)

plt.savefig('validation_accuracy_token_biochem_700.png')
plt.show()

plt.figure(figsize=(10, 6))
for file, label in zip(files, custom_labels):
    df = pd.read_csv(file)
    plt.plot(df['epoch'], df['accuracy_arc'], label=label)

plt.xlabel('Epoch')
plt.ylabel('Accuracy ARC')
plt.title('Validation Accuracy ARC over Epochs')
plt.legend(loc='best')
plt.grid(True)
plt.xticks([100, 300, 500, 700, 700])
plt.xlim(0, 700)
plt.ylim(0.88, 0.95)

plt.savefig('validation_accuracy_arc_biochem_700.png')
plt.show()

plt.figure(figsize=(10, 6))
for file, label in zip(files, custom_labels):
    df = pd.read_csv(file)
    plt.plot(df['epoch'], df['accuracy_brc'], label=label)

plt.xlabel('Epoch')
plt.ylabel('Accuracy BRC')
plt.title('Validation Accuracy BRC over Epochs')
plt.legend(loc='best')
plt.grid(True)
plt.xticks([100, 300, 500, 700, 700])
plt.xlim(0, 700)
plt.ylim(0.74, 0.90)

plt.savefig('validation_accuracy_brc_biochem_700.png')
plt.show()

print("All accuracy graphs have been saved as PNG files.")
