import pandas as pd

dataset_train_retroformer_readretro = pd.read_csv('/home/lmartins/READRetro/scripts/singlestep_eval/retroformer/biochem/data/raw_train.csv')

dataset_train_retroformer_readretro = dataset_train_retroformer_readretro[['id', 'class', 'reactants>reagents>production']]

plantcyc_dataset = pd.read_csv('/home/lmartins/Retroformer/Retroformer_Thesis/data_plantcyc/raw_train.csv')

merge_dataset = pd.concat([dataset_train_retroformer_readretro, plantcyc_dataset], ignore_index=True)

merge_dataset.to_csv('BioChem+PlantCyc.csv', index=False)