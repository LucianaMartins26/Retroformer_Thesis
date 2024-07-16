import pickle
import csv
import numpy as np

with open('vanilla_bs_top4_generation_untyped.pk', 'rb') as file:
    data = pickle.load(file)

with open('vanilla_bs_top4_generation_untyped.txt', 'w') as file:
    file.write(str(data[0]))
    file.write('\n')
    file.write(str(data[1]))

""" if isinstance(data[1], list) and all(isinstance(item, np.ndarray) for item in data[1]):
    second_element_flat = [item for sublist in data[1] for item in sublist]
else:
    second_element_flat = data[1]

approach = 'truncate'  # ou 'pad'

if approach == 'truncate':
    # Truncar as listas para o comprimento da lista mais curta
    min_length = min(len(data[0]), len(second_element_flat))
    first_list = data[0][:min_length]
    second_list = second_element_flat[:min_length]
elif approach == 'pad':
    # Preencher a lista mais curta com valores nulos até o comprimento da mais longa
    max_length = max(len(data[0]), len(second_element_flat))
    first_list = data[0] + [''] * (max_length - len(data[0]))
    second_list = second_element_flat + [''] * (max_length - len(second_element_flat))

combined_data = list(zip(first_list, second_list))

with open('vanilla_bs_top3_generation_untyped.csv', 'w', newline='') as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(['Column1', 'Column2'])
    writer.writerows(combined_data)
print("Dados convertidos para 'vanilla_bs_top3_generation_untyped.csv'") """
