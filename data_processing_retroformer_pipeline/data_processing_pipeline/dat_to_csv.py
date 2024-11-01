import pandas as pd
import luigi

def get_record(filename):
    all_records = []
    with open(filename, 'r', encoding='ISO-8859-1') as datafile:
        record = {}
        for line in datafile:
            line = line.strip()
            if line.startswith("//"):
                all_records.append(record)
                record = {}
            elif line and not line.startswith("#"):
                if ' - ' in line:
                    key, value = line.split(' - ', 1)
                    if key in record:
                        record[key].append(value)
                    else:
                        record[key] = [value]
    return all_records

class CreateDataFrame(luigi.Task):

    def input(self):
        return luigi.LocalTarget('data_processing_pipeline/data/reactions.dat')

    def output(self):
        return luigi.LocalTarget('data_processing_pipeline/res/reactions.csv')

    def run(self):
        records = get_record(self.input().path)

        data_dict = {
            "ID": [],
            #"PATHWAY": [],
            "REACTANTS": [],
            "PRODUCTS": [],
            "REACTION-DIRECTION": []
        }

        for record in records:
            unique_id = record.get("UNIQUE-ID", [])
            #pathway = record.get("IN-PATHWAY", [])
            reaction_direction = record.get("REACTION-DIRECTION", [])
            left = record.get("LEFT", [])
            right = record.get("RIGHT", [])
            
            left_combined = ", ".join(left) if left else ""
            right_combined = ", ".join(right) if right else ""
            
            data_dict["ID"].append(unique_id[0] if unique_id else "")
            #data_dict["PATHWAY"].append(pathway[0] if pathway else "")
            data_dict["REACTANTS"].append(left_combined)
            data_dict["PRODUCTS"].append(right_combined)
            data_dict["REACTION-DIRECTION"].append(reaction_direction[0] if reaction_direction else "")

        df = pd.DataFrame(data_dict)
        df.to_csv(self.output().path, index=False)
