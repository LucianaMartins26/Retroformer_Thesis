import pandas as pd
import matplotlib.pyplot as plt

files = [
    'log_training_file_TL700.csv',
    'log_training_file_TL700_LR10x.csv',
    'log_training_file_TL700_LR100x.csv',
    'log_training_file_TL700_LR1000x.csv',
    'log_training_file_TL700_ExtraLayer.csv'
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
    plt.plot(df['Epoch'], df['Loss'], label=label)

plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Loss over Epochs')
plt.legend(loc='best')
plt.grid(True)

plt.savefig('loss_epochs_behaviour_biochem_700.png')

plt.show()