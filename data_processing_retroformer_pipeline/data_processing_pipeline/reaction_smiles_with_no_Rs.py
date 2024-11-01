import pandas as pd
from data_processing_pipeline.compounds_to_smiles import CompoundToSmiles
import luigi
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

class ReactionSmilesWithNoRs(luigi.Task):

    def requires(self):
        return CompoundToSmiles()

    def input(self):
        return luigi.LocalTarget('data_processing_pipeline/res/reactions_SM_RD_SMILES.csv')

    def output(self):
        return luigi.LocalTarget('data_processing_pipeline/res/ready_data_no_atom_mapped.csv')

    def reaction_smiles(self, dataset):
        dataset.insert(5, 'REACTION-SMILES', dataset['REACTANTS'] + '>>' + dataset['PRODUCTS'])
        return dataset

    def replace_radicals(self, dataset):
        dataset['REACTION-SMILES'] = dataset['REACTION-SMILES'].apply(transformed_smiles)
        return dataset

    def run(self):
        dataset_path = self.input().path
        dataset = pd.read_csv(dataset_path)

        dataset = self.reaction_smiles(dataset)
        dataset = self.replace_radicals(dataset)

        dataset = dataset.dropna(subset=['REACTION-SMILES'])
        dataset = dataset[~dataset['REACTION-SMILES'].str.contains('*', regex=False, na=False)]
        dataset.reset_index(drop=True, inplace=True)

        output_path = self.output().path
        dataset.to_csv(output_path, index=False)
        