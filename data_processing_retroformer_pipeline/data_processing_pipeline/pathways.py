import pandas as pd
import luigi
import pickle

from data_processing_pipeline.dat_to_csv import get_record, CreateDataFrame

def find_pathway_id_for_rxn(all_records, rxn):
    for record in all_records:
        for key, values in record.items():
            for value in values:
                if rxn in value:
                    return record.get('UNIQUE-ID', [None])[0]
    return None

class Pathways(luigi.Task):

    def requires(self):
        return CreateDataFrame()

    def input(self):
        return {'Dataset' : luigi.LocalTarget('data_processing_pipeline/res/reactions.csv'), 
                'Pathways' : luigi.LocalTarget('data_processing_pipeline/data/pathways.dat'),
                'Reactions' : luigi.LocalTarget('data_processing_pipeline/data/reactions.dat')}

    def output(self):
        return luigi.LocalTarget('data_processing_pipeline/res/reactions_with_pathways.csv')

    def run(self):
        pathways_records = get_record(self.input()['Pathways'].path)
        reactions_records = get_record(self.input()['Reactions'].path)
        dataset = pd.read_csv(self.input()['Dataset'].path)

        pathways = dataset['ID'].map(lambda x: find_pathway_id_for_rxn(pathways_records, x))
        dataset.insert(1, 'PATHWAYS', pathways)
        
        for index, row in dataset.iterrows():
            if pd.isna(row['PATHWAYS']):
                dataset.at[index, 'PATHWAYS'] = 'UNK'
                        
        dataset.to_csv(self.output().path, index=False)