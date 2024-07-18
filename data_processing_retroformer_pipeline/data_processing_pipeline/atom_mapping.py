import pandas as pd
from rxnmapper import BatchedMapper
import luigi
from data_processing_pipeline.reaction_smiles_with_no_Rs import ReactionSmilesWithNoRs

class AtomMapping(luigi.Task):

    def requires(self):
        return ReactionSmilesWithNoRs()

    def input(self):
        return luigi.LocalTarget('data_processing_pipeline/res/ready_data_no_atom_mapped.csv')

    def output(self):
        return luigi.LocalTarget('data_processing_pipeline/res/ready_data.csv')

    def run(self):
        dataset_path = self.input().path
        dataset = pd.read_csv(dataset_path)

        bm = BatchedMapper(batch_size=5)

        results = list(bm.map_reactions(list(dataset['REACTION-SMILES'])))
        dataset.insert(6, 'REACTION-SMILES-AM', results)
        dataset = dataset[dataset['REACTION-SMILES-AM'] != '>>']
        dataset = dataset.reset_index(drop=True)

        dataset.to_csv(self.output().path, index=False)

        