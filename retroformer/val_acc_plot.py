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

plt.figure(figsize=(10, 6))
for file, label in zip(files, custom_labels):
    df = pd.read_csv(file)
    plt.plot(df['epoch'], df['accuracy_token'], label=label)

plt.xlabel('Epoch')
plt.ylabel('Accuracy Token')
plt.title('Validation Accuracy Token over Epochs')
plt.legend(loc='best')
plt.grid(True)
plt.xticks([100, 300, 500, 700, 1000])
plt.xlim(0, 1000)
plt.ylim(0.825, 0.975)

plt.savefig('validation_accuracy_token.png')
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
plt.xticks([100, 300, 500, 700, 1000])
plt.xlim(0, 1000)
plt.ylim(0.875, 0.925)

plt.savefig('validation_accuracy_arc.png')
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
plt.xticks([100, 300, 500, 700, 1000])
plt.xlim(0, 1000)
plt.ylim(0.78, 0.85)

plt.savefig('validation_accuracy_brc.png')
plt.show()

print("All accuracy graphs have been saved as PNG files.")
