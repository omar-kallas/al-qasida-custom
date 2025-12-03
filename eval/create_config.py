import json # consider switching to yaml 
from maps import DIALECTS8

yn2bool = {
    "y": True,
    "Y": True,
    "n": False,
    "N": False
}

config = {}

# (1) load_model_type 

load_model_choices = {
    "non-pipeline", "gated-pipeline", "ungated-pipeline"
}
pipeline_or_not = ""
while pipeline_or_not not in yn2bool:
    pipeline_or_not = input(
        "Should your model be loaded using `transformers.pipeline`? (Y/n) "
    )
if not yn2bool[pipeline_or_not]:
    load_model_type = "non-pipeline"
else:
    gated_or_not = ""
    while gated_or_not not in yn2bool:
        gated_or_not = input(
            "Is your model gated on HuggingFace? (Y/n) "
        )
    if yn2bool[gated_or_not]:
        load_model_type = "gated-pipeline"
    else:
        load_model_type = "ungated-pipeline"
assert load_model_type in load_model_choices 
config["load_model_type"] = load_model_type 

# (2) hf_name
hf_name = input("What is the HuggingFace ID for this model? ")
config['hf_name'] = hf_name 

# (3) prefixes 
print("Now add dialects you want to support.")
possible_dialects = set(DIALECTS8)
choice_str = " | ".join(DIALECTS8)
dialects = []
next_dialect = "" 
while True:
    next_dialect = input(
        f"Please enter dialect #{len(dialects) + 1} or 'stop' to end:\n"\
        f"choices: [ {choice_str} | stop ]\n"
    )
    if next_dialect in possible_dialects:
        dialects.append(next_dialect)
    elif next_dialect == 'stop': 
        break
    else:
        print("*Dialect not recognized. Skipping.*")
dial2prefix = {}
for dialect in dialects:
    prefix_fn = input(
        f"What file contains the prefix for the {dialect} dialect?\n"\
        "(empty input will give an empty prefix)\n"
    )
    if prefix_fn:
        with open(prefix_fn, 'r') as f:
            prefix = f.read() 
        dial2prefix[dialect] = prefix 
    else:
        dial2prefix[dialect] = ""
config['prefixes'] = dial2prefix 

# (-1) output file 
out_file = input("What JSON file do you want to write this config to? ")

with open(out_file, 'w') as f:
    json.dump(config, f, indent=4, ensure_ascii=False)
print("Written", out_file)

