import pandas as pd
import re
import luigi

from data_processing_pipeline.secondary_metabolism_and_reaction_direction import SecondaryMetabolismAndReactionDirection

def get_compounds(filepath):
    compounds = {}
    with open(filepath, 'r', encoding='ISO-8859-1') as file:
        lines = file.readlines()
        unique_id = None
        smiles = None
        for line in lines:
            line = line.strip()
            if line.startswith('UNIQUE-ID'):
                unique_id = line.split(' - ')[1].strip()
            elif line.startswith('SMILES'):
                smiles = line.split(' - ')[1].strip()
            if unique_id and smiles:
                compounds[unique_id] = smiles
                unique_id = None
                smiles = None
    return compounds

def valid_smiles(smiles):
    pattern = r'^[A-Za-z0-9@\.\+\-\[\]\(\)\\/#=]+$'
    return re.match(pattern, smiles) is not None

def smiles_replacement(compound_list, compound_dict):
    result = []
    for compound in compound_list:
        if compound in compound_dict and valid_smiles(compound_dict[compound]):
            smiles = compound_dict[compound]
            if smiles == 'S1212':
                result.append('[S]')
            else:
                result.append(smiles)
        else:
            pass
    return '.'.join(result)

class CompoundToSmiles(luigi.Task):
    
    def requires(self):
        return SecondaryMetabolismAndReactionDirection()

    def input(self):
        return {'Compounds' : luigi.LocalTarget('data_processing_pipeline/data/compounds.dat'), 
                'Dataset' : luigi.LocalTarget('data_processing_pipeline/res/reactions_SM_RD.csv')}

    def output(self):
        return luigi.LocalTarget('data_processing_pipeline/res/reactions_SM_RD_SMILES.csv')

    def run(self):
        dataset = self.input()['Dataset'].path
        dataset = pd.read_csv(dataset)
        compound_dict = self.input()['Compounds'].path
        compound_dict = get_compounds(compound_dict)

        for index, row in dataset.iterrows():
            if isinstance(row['REACTANTS'], str):
                reactants = [x.strip() for x in row['REACTANTS'].split(',')]
                dataset.at[index, 'REACTANTS'] = smiles_replacement(reactants, compound_dict)
            if isinstance(row['PRODUCTS'], str):
                products = [x.strip() for x in row['PRODUCTS'].split(',')]
                dataset.at[index, 'PRODUCTS'] = smiles_replacement(products, compound_dict)

        dataset.to_csv(self.output().path, index=False)