import os
from datasets import load_dataset

SPLIT = 'dev' # can also be devtest, as in original AL-QASIDA paper
# Do NOT set to devtest for AMIYA shared task

DA_ISO_CODES = [ # Which Dialectal Arabic varieties to include
    'ary', # Moroccan Arabic 
    'arz', # Egyptian Arabic 
    'ajp', # South Levantine (Palestinian) Arabic (NOTE : It seems this was removed in the most recent versions of FLORES? Need to look into this....)
    'apc', # North Levantine (Syrian) Arabic
    'ars', # Najdi (Saudi) Arabic 
]

def get_lines(iso, script, split=SPLIT): 
    # Example: get_lines('arz', 'Arab')
    flores_code = f"{iso}_{script}"
    data = load_dataset(
        "facebook/flores", 
        flores_code, 
        split=split, 
        trust_remote_code=True
    )
    lines = [datum['sentence'].strip() + '\n' for datum in data] 
    return lines 

def main(verbose=True, split=SPLIT):
    for iso in DA_ISO_CODES + ['arb', 'eng']: 
        # Retrieve data
        script = "Latn" if iso == "eng" else "Arab"
        flores_code = f"{iso}_{script}"
        lines = get_lines(iso, script, split=split)
        # Write file
        out_dir = f"bitexts/flores200_dataset/{split}"
        if not os.path.exists(out_dir):
            os.makedirs(out_dir)
        out_file = os.path.join(out_dir, f"{flores_code}.{split}")
        with open(out_file, 'w') as f:
            f.writelines(lines)
        if verbose:
            print("Written", out_file)
    

if __name__ == "__main__":
    main()
    main(split="dev")
