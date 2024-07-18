import pandas as pd
import luigi
import pickle

from data_processing_pipeline.pathways import Pathways

def is_secondary_metabolism(pathway, secondary_metabolites_pathways):
    return pathway in secondary_metabolites_pathways

class SecondaryMetabolismAndReactionDirection(luigi.Task):

    def requires(self):
        return Pathways()

    def input(self):
        return {'SMP_file' : luigi.LocalTarget('data_processing_pipeline/data/secondary_metabolites_pathway.pkl'), 
                'Dataset' : luigi.LocalTarget('data_processing_pipeline/res/reactions_with_pathways.csv')}

    def output(self):
        return luigi.LocalTarget('data_processing_pipeline/res/reactions_SM_RD.csv')

    def secondary_metabolism_identification(self, dataset_path):
        with open(self.input()['SMP_file'].path, 'rb') as f:
            secondary_metabolites_pathways = pickle.load(f)

        df = pd.read_csv(dataset_path)
        secondary_metabolism = df['PATHWAYS'].apply(lambda x: is_secondary_metabolism(x, secondary_metabolites_pathways))

        df.insert(2, 'SECONDARY-METABOLISM', secondary_metabolism)

        return df

    def reaction_direction_adjustments(self, dataset):
        new_rows = []

        dataset['REACTION-DIRECTION'] = dataset['REACTION-DIRECTION'].apply(
            lambda direction: 'LEFT-TO-RIGHT' if direction in ['PHYSIOL-LEFT-TO-RIGHT', 'IRREVERSIBLE-LEFT-TO-RIGHT']
            else 'RIGHT-TO-LEFT' if direction in ['PHYSIOL-RIGHT-TO-LEFT', 'IRREVERSIBLE-RIGHT-TO-LEFT']
            else direction
        )

        for index, row in dataset.iterrows():
            if row['REACTION-DIRECTION'] == 'REVERSIBLE':
                dataset.at[index, 'REACTION-DIRECTION'] = 'LEFT-TO-RIGHT'
                
                new_row = row.copy()
                new_row['REACTANTS'] = dataset.at[index, 'PRODUCTS']
                new_row['PRODUCTS'] = dataset.at[index, 'REACTANTS']
                new_row['REACTION-DIRECTION'] = 'RIGHT-TO-LEFT'
                new_rows.append(new_row)

        dataset = pd.concat([dataset, pd.DataFrame(new_rows)], ignore_index=True)
        return dataset

    def run(self):
        dataset_path = self.input()['Dataset'].path
        dataset = self.secondary_metabolism_identification(dataset_path)
        dataset = self.reaction_direction_adjustments(dataset)
        dataset.to_csv(self.output().path, index=False)