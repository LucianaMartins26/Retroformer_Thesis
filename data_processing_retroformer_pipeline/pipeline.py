import luigi

from data_processing_pipeline.retroformer_training_format import RetroformerFormat

import sys

sys.path.append("../data_processing_pipeline")

class PlantCycPipeline(luigi.WrapperTask):

    def requires(self):
        return RetroformerFormat()