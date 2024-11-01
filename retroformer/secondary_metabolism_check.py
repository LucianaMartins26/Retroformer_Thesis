from rdkit import Chem
import pandas as pd


df = pd.read_csv('../../../data_processing_retroformer_pipeline/data_processing_pipeline/res/ready_data.csv')

selected_columns = ['ID', 'PATHWAYS', 'REACTION-SMILES-AM']

new_names = {
    'ID': 'id',
    'PATHWAYS': 'class',
    'REACTION-SMILES-AM': 'reactants>reagents>production'
}

retroformer_format_dataset = df[selected_columns].rename(columns=new_names)

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

i = 0
for idx in idx_to_drop:
    if df.at[idx, 'SECONDARY-METABOLISM']:
        i += 1

message = 'There are {} invalid SMILES that belong to secondary metabolism pathways between {}.'.format(i, len(idx_to_drop))

with open('InvalidSMILES.txt', 'w') as file:
    file.write(message)