"""
This file defines the classes used to organize input and 
output data in ./evaluator.py (imported in ./evaluator.py).

If called on its own, it can also organize the output data
after evaluation. (This is primarily because the original 
version of ./evaluator.py didn't do output data organization, 
but now it is just part of the program.)
"""

import os, glob, argparse, pdb
import pickle as pkl 
import pandas as pd 

GENRE2STR = {
    "btec/madar26": "BTEC",
    "music/habibi": "HABIBI",
    "tweets/nadi2023": "TWEET",
    "wiki/flores": "FLORES",
    "hehe": "Cohere",
    "okapi": "Okapi",
    "sharegpt": "ShareGPT"
}
TASK2SUBDIRS = {
    "monolingual": [
        "btec/madar26",
        "music/habibi",
        "tweets/nadi2023",
        "wiki/flores"
    ],
    "crosslingual": [
        "hehe",
        "okapi", 
        "sharegpt"
    ],
    "mt": [
        "wiki/flores",
        "btec/madar26"
    ]
}

class InDataOrganizer():
    def __init__(self, data_dir, task="monolingual", test=False):
        self.parent_dir = data_dir 
        self.subdirs = TASK2SUBDIRS[task]
        self.task = task 
        self.test = test

    def get_prompts(self, csv_path):
        try:
            df = pd.read_csv(csv_path) 
        except:
            print("Could not process", csv_path, "!!!!")
            raise
        if self.test:
            df = df.head(5)
        return [p for p in df['prompt']] 
    
    def get_refs(self, csv_path):
        df = pd.read_csv(csv_path) 
        if self.test:
            df = df.head(5)
        return [c for c in df['completion']] 
    
    def get_dialect(self, csv_path): 
        assert csv_path.endswith(".csv"), f"Wrong CSV name: {csv_path}"
        return os.path.split(csv_path)[-1][:-4]
    
    def organize_prompts(self, mt_refs=False):
        if mt_refs:
            get_texts_fcn=self.get_refs 
        else:
            get_texts_fcn=self.get_prompts
        # Get organized 
        organization = {subdir: {} for subdir in self.subdirs} 
        for subdir in self.subdirs: 
            csv_fns = glob.glob(os.path.join(self.parent_dir, subdir, '*.csv'))
            for csv_fn in csv_fns: 
                dialect = self.get_dialect(csv_fn)
                csv_prompts = get_texts_fcn(csv_fn)
                organization[subdir][dialect] = csv_prompts 
        return organization

class OutDataOrganizer():
    def __init__(self, data_dict, llm, task, reports_dir, layer="", coef=""):
        # data_dict maps genre -> dialect -> metric -> mean score
        self.data_dict = data_dict  
        self.llm = llm 
        self.task = task 
        self.reports_dir = reports_dir
        if layer and coef:
            self.dirname = f"{self.llm}_l{layer}_c{coef}_{self.task}" 
        else:
            self.dirname = f"{self.llm}_{self.task}" 
        # We want an organization like this:
        # - llm and task in dir name (e.g. command_r_monolingual)
        # - genre and dialect in filename 
        #       (e.g. DialectID_BTEC_dza_metrics.csv)
        # - metrics are column names in single-row csv 
    
    def organize_data(self): 
        dirname = self.dirname
        out_dir = os.path.join(self.reports_dir, dirname) 
        if not os.path.exists(out_dir):
            os.makedirs(out_dir)
        id_2letters = "MT" if self.task == "mt" else "ID"
        for genre in self.data_dict:
            for dialect in self.data_dict[genre]:
                genre_str = GENRE2STR[genre]
                csv_file = f"Dialect{id_2letters}_{genre_str}"\
                    f"_{dialect}_metrics.csv"
                csv_data = {
                    metric: [
                        float(self.data_dict[genre][dialect][metric])
                    ] for metric in self.data_dict[genre][dialect]
                }
                csv_df = pd.DataFrame(csv_data)
                out_path = os.path.join(out_dir, csv_file)
                csv_df.to_csv(out_path, index=False)
                print(out_path, "written")

if __name__  == "__main__":

    parser = argparse.ArgumentParser() 
    parser.add_argument("--data-pkl", required=True, type=str)
    parser.add_argument("--llm", default="llama", type=str)
    parser.add_argument("--task", default="monolingual", type=str)
    parser.add_argument("--out-dir", default="../llm_outputs", type=str)
    args = parser.parse_args() 

    with open(args.data_pkl, 'rb') as f:
        data_dict = pkl.load(f)

    out_organizer = OutDataOrganizer(
        data_dict, args.llm, args.task, args.out_dir
    )
    out_organizer.organize_data()
