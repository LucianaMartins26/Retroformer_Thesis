import pandas as pd
import matplotlib.pyplot as plt

for number in range(0, 5):
    file = 'log_training_file_cuda{}.csv'.format(number)
    cuda_code = file.rsplit('_', 1)[-1]

    df = pd.read_csv(file)

    plt.figure(figsize=(10, 6))

    loss_columns = [col for col in df.columns if 'Loss' in col]

    for col in loss_columns:
        plt.plot(df['Epoch'], df[col], label=col)

    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Comportamento das Losses ao longo das Epochs - {}'.format(cuda_code))
    plt.legend()

    plt.savefig('loss_epochs_cuda{}.png'.format(number))

    plt.show()