import luigi
from pipeline import PlantCycPipeline

luigi.build([PlantCycPipeline()], workers=1, scheduler_host='127.0.0.1',
            scheduler_port=8083, local_scheduler=True)