import pandas as pd
from rdkit import Chem

dataset = pd.read_csv('../data_processing_retroformer_pipeline/data_processing_pipeline/res/ready_data.csv')

selected_columns = ['ID', 'PATHWAYS', 'REACTION-SMILES-AM']

new_names = {
    'ID': 'id',
    'PATHWAYS': 'class',
    'REACTION-SMILES-AM': 'reactants>reagents>production'
}

dataset = dataset[dataset['SECONDARY-METABOLISM'] == True]

retroformer_format_dataset = dataset[selected_columns].rename(columns=new_names)

invalid_r = set()
invalid_p = set()

for idx, row in retroformer_format_dataset.iterrows():
    r, p = row['reactants>reagents>production'].split('>>')
    
    reactants = r.split('.')
    for reac in reactants:
        if not Chem.MolFromSmiles(reac):
            invalid_r.add(idx)

    products = p.split('.')
    for prod in products:
        if not Chem.MolFromSmiles(prod):
            invalid_p.add(idx)

idx_to_drop = list(invalid_r.union(invalid_p))

retroformer_format_dataset = retroformer_format_dataset.drop(idx_to_drop)

retroformer_format_dataset.to_csv('plantcyc_dataset_sm_only.csv', index=False)