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

plt.figure(figsize=(10, 6))

for file, label in zip(files, custom_labels):
    df = pd.read_csv(file)
    plt.plot(df['Epoch'], df['Loss'], label=label)

plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Loss over Epochs')
plt.legend(loc='best')
plt.grid(True)

plt.xticks([100, 300, 500, 700, 1000])

plt.savefig('loss_epochs_behaviour.png')

plt.show()