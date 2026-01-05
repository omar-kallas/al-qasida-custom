from format_xtext import *
import os

DEVTEST = "dev" # Was "test" in original AL-QASIDA paper
# Do NOT set to "test" for AMIYA shared task
if DEVTEST == "dev":
    MADAR_SPLIT = "corpus-6-test-corpus-26-dev"
    FLORES_SPLIT = "dev"
elif DEVTEST == "test":
    MADAR_SPLIT = "corpus-6-test-corpus-26-test"
    FLORES_SPLIT = "devtest"
else:
    raise NotImplementedError("DEVTEST must be 'dev' or 'test'")

SRC2INFO = {
    "flores": {
        "bitext": True,
        "temp": lambda x: "bitexts/flores200_dataset/{}/{}.{}".format(
            FLORES_SPLIT, x, FLORES_SPLIT
        ),
        "from_code": lambda x: COUNTRY2ISO[x], 
        "to_fn_version": lambda x: x + "_Latn" if x == 'eng' else x + "_Arab",
        "txt_keys": ["", ""],
        "lang_type": "iso",
        "name": "FLORES-200",
        "genre": "wiki",
        "c2filter_str": {c: "" for c in COUNTRY2ISO},
        "countries": [c for c in COUNTRY2ISO]
    },
    "flores-dev": {
        "bitext": True,
        "temp": lambda x: "bitexts/flores200_dataset/dev/{}.dev".format(x),
        "from_code": lambda x: COUNTRY2ISO[x], 
        "to_fn_version": lambda x: x + "_Latn" if x == 'eng' else x + "_Arab",
        "txt_keys": ["", ""],
        "lang_type": "iso",
        "name": "FLORES-200-dev",
        "genre": "wiki",
        "c2filter_str": {c: "" for c in COUNTRY2ISO},
        "countries": [c for c in COUNTRY2ISO]
    },
    "madar26": {
        "bitext": True,
        "temp": lambda x: "bitexts/MADAR.Parallel-Corpora-Public-Version1.1-25MAR2021/MADAR_Corpus/MADAR.corpus.{}.tsv".format(x),
        "from_code": lambda x: COUNTRY2CITY[x], 
        "to_fn_version": lambda x: CITY2NAME[x.upper()],
        "txt_keys": ["sent", "sent"],
        "lang_type": "city",
        "name": "MADAR-26",
        "genre": "btec",
        "c2filter_str": {c: f"split == '{MADAR_SPLIT}'" for c in COUNTRY2CITY},
        "countries": [c for c in COUNTRY2CITY]
    },
    "nadi2023": {
        "bitext": False,
        "temp": lambda x: "monotexts/NADI2023_Release_Train/Subtask1/NADI2023_Subtask1_DEV.tsv",
        "from_code": lambda x: COUNTRY2NAME[x], 
        "to_fn_version": lambda x: "", 
        "txt_keys": ["content", ""],
        "lang_type": "name",
        "name": "NADI-2023",
        "genre": "tweets",
        "c2filter_str": {c: f"label == '{COUNTRY2NAME[c]}'" for c in COUNTRY2NAME},
        "countries": [c for c in COUNTRY2NAME]
    },
    "habibi": {
        "bitext": False,
        "temp": lambda x: "monotexts/HABIBI/arabicLyrics_cleaned.csv",
        "from_code": lambda x: COUNTRY2NAME[x], 
        "to_fn_version": lambda x: x.split(' ')[0], 
        "txt_keys": ["Lyrics", ""],
        "lang_type": "name",
        "name": "HABIBI",
        "genre": "music",
        "c2filter_str": {c: f"SingerNationality == '{COUNTRY2NAME[c]}'" for c in COUNTRY2NAME},
        "countries": [c for c in COUNTRY2NAME]
    }
}
