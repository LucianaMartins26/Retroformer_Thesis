import pandas as pd
import re

def transformed_smiles(smiles):
    transformed_reactants = []
    transformed_products = []

    if isinstance(smiles, str):
        reactants_and_products = smiles.split(">>")
        
        for reactant in reactants_and_products[0].split("."):
            reactant = re.sub(r'\[R(\d*?)\]', '*', reactant)
            transformed_reactants.append(reactant)

        for product in reactants_and_products[1].split("."):
            product = re.sub(r'\[R(\d*?)\]', '*', product)
            transformed_products.append(product)

        transformed_smiles = ".".join(transformed_reactants) + ">>" + ".".join(transformed_products)

        return transformed_smiles


dataset = pd.read_csv('../data_processing_retroformer_pipeline/data_processing_pipeline/res/reactions_SM_RD_SMILES.csv')
dataset.insert(5, 'REACTION-SMILES', dataset['REACTANTS'] + '>>' + dataset['PRODUCTS'])
dataset['REACTION-SMILES'] = dataset['REACTION-SMILES'].apply(transformed_smiles)
dataset = dataset.dropna(subset=['REACTION-SMILES'])
dataset = dataset[dataset['REACTION-SMILES'].str.contains('*', regex=False, na=False)]

i = 0
for index, row in dataset.iterrows():
    if row['SECONDARY-METABOLISM']:
        i += 1

message = 'There are {} reaction SMILES with radicals that belong to secondary metabolism pathways between {}.'.format(i, len(dataset))

with open('RadicalsSMILES.txt', 'w') as file:
    file.write(message)