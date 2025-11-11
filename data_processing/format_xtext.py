import argparse
import os
import json
import random 
import pdb 

import pandas as pd

random.seed(sum(bytes(b'cohere')))

"""
Bitext csv columns:
,prompt,completion,source,target_language,source_language,language

Monotext csv columns:
,prompt,language,source
"""

MT_PROMPT_TEMP = 'Translate from {src_lang} into {tgt_lang}. Output only the translation, '\
                 'do NOT output anything else before nor after it.\n{src_txt}\nTranslation:'

COUNTRY2DIAL = {
    "dza": "Algerian Arabic",
    "bhr": "Bahrani Arabic",
    "egy": "Egyptian Arabic",
    "irq": "Iraqi Arabic",
    "jor": "Jordanian Arabic",
    "kwt": "Kuwaiti Arabic",
    "lbn": "Lebanese Arabic",
    "lby": "Libyan Arabic",
    "mar": "Moroccan Arabic",
    "omn": "Omani Arabic",
    "pse": "Palestinian Arabic",
    "qat": "Qatari Arabic",
    "sau": "Saudi Arabic",
    "sdn": "Sudanese Arabic",
    "syr": "Syrian Arabic",
    "tun": "Tunisian Arabic",
    "are": "Emirati Arabic",
    "yem": "Yemeni Arabic",
    "msa": "Modern Standard Arabic",
    "eng": "English"
}
DIAL2COUNTRY = {COUNTRY2DIAL[key]: key for key in COUNTRY2DIAL}
COUNTRY2NAME = {
    "dza": "Algeria",
    "bhr": "Bahrain",
    "egy": "Egypt",
    "irq": "Iraq",
    "jor": "Jordan",
    "kwt": "Kuwait",
    "lbn": "Lebanon",
    "lby": "Libya",
    "mar": "Morocco",
    "omn": "Oman",
    "pse": "Palestine",
    "qat": "Qatar",
    "sau": "Saudi_Arabia",
    "sdn": "Sudan",
    "syr": "Syria",
    "tun": "Tunisia",
    "are": "UAE",
    "yem": "Yemen"
}
NAME2COUNTRY = {COUNTRY2NAME[key]: key for key in COUNTRY2NAME}
CITY2COUNTRY = {
    "ALE": "syr",
    "ALX": "egy",
    "ALG": "dza",
    "AMM": "jor",
    "ASW": "egy",
    "BAG": "irq",
    "BAS": "irq",
    "BEI": "lbn",
    "BEN": "lby",
    "CAI": "egy",
    "DAM": "syr",
    "DOH": "qat",
    "FES": "mar",
    "JED": "sau",
    "JER": "pse",
    "KHA": "sdn",
    "MSA": "msa",
    "msa": "msa",
    "MOS": "irq",
    "MUS": "omn",
    "RAB": "mar",
    "RIY": "sau",
    "SAL": "jor",
    "SAN": "yem",
    "SFX": "tun",
    "TRI": "lby",
    "TUN": "tun",
    "ENG": "eng"
}
COUNTRY2CITY = {
    "dza": "ALG",
    "jor": "AMM",
    "irq": "BAG",
    "lbn": "BEI",
    "egy": "CAI",
    "syr": "DAM",
    "qat": "DOH",
    "mar": "FES",
    "pse": "JER",
    "sdn": "KHA",
    "msa": "MSA",
    "omn": "MUS",
    "sau": "RIY",
    "yem": "SAN",
    "lby": "TRI",
    "tun": "TUN",
    "eng": "ENG",
    "msa": "MSA"
}
CITY2NAME = {
    "ALE": "Aleppo",
    "ALX": "Alexandria",
    "ALG": "Algiers",
    "AMM": "Amman",
    "ASW": "Aswan",
    "BAG": "Baghdad",
    "BAS": "Basra",
    "BEI": "Beirut",
    "BEN": "Benghazi",
    "CAI": "Cairo",
    "DAM": "Damascus",
    "DOH": "Doha",
    "FES": "Fes",
    "JED": "Jeddah",
    "JER": "Jerusalem",
    "KHA": "Khartoum",
    "MSA": "MSA",
    "msa": "MSA",
    "MOS": "Mosul",
    "MUS": "Muscat",
    "RAB": "Rabat",
    "RIY": "Riyadh",
    "SAL": "Salt",
    "SAN": "Sanaa",
    "SFX": "Sfax",
    "TRI": "Tripoli",
    "TUN": "Tunis",
    "ENG": "English"
}
ISO2COUNTRY = {
    "arz": "egy",
    "acm": "irq",
    "ajp": "pse",
    "ary": "mar",
    "ars": "sau",
    "apc": "syr",
    "aeb": "tun",
    "acq": "yem",
    "arb": "msa",
    "eng": "eng"
}
COUNTRY2ISO = {ISO2COUNTRY[key]: key for key in ISO2COUNTRY}
# GENRE2MACROGENRE = {
#     "wiki": "wiki",
#     "tweets": "utt",
#     "btec": "utt",
#     "music": "utt"
# }
GENRE2MACROGENRE = {
    "wiki": "all",
    "tweets": "all",
    "btec": "all",
    "music": "all"
}

BILIMIT = 200
MONOLIMIT = 100

SEED = sum(bytes(b'mena'))

def limit_df(df: pd.DataFrame, limit: int) -> pd.DataFrame:
    df = df.sample(frac=1, random_state=SEED, ignore_index=True)
    return df.truncate(before=0, after=limit-1)

class FormatData():
    """
    """
    def __init__(self, config: dict):
        self.xtext = config['xtext']
        self.in_files = config['in_files'] 
        self.ftype = config['ftype']
        self.src_txt_key = config['src_txt_key']
        self.lang_type = config['lang_type']
        self.source = config['source']
        self.out_csv = config['out_csv']
        self.genre = config['genre']
        self.filter_str = config['filter_str']
        self.lang_pair = ""

        self.iso2country = ISO2COUNTRY 
        self.city2country = CITY2COUNTRY 
        self.country2dial = COUNTRY2DIAL
        self.name2country = NAME2COUNTRY

        self.src_lang = self.convert_lang(config['src_lang'], self.lang_type)
    
    def __str__(self):
        stemp = "{xtext}-{lang_pair}-{source}"
        return stemp.format(xtext=self.xtext, lang_pair=self.lang_pair.replace(' ', '_'), source=self.source)

    def convert_lang(self, lang_str, lang_type):
            """
            We want the lang to be the country code here!
            """
            if lang_type == "country":
                return lang_str
            elif lang_type == "city":
                return self.city2country[lang_str]
            elif lang_type == "iso":
                return self.iso2country[lang_str]
            elif lang_type == "name":
                return self.name2country[lang_str]
            else:
                raise NotImplementedError(f"Lang str type not supported: {lang_type}")
    
    def file2sents(self, fn, key, ftype) -> list:
        if ftype == "txt":
            with open(fn, 'r') as f:
                sents = [s.strip() for s in f.readlines()]
            return sents 
        elif ftype == "csv":
            df = pd.read_csv(fn)
            if self.filter_str:
                df = df.query(self.filter_str)
            sents = df[key].values.tolist()
            return sents 
        elif ftype == "tsv":
            df = pd.read_table(fn)
            if self.filter_str:
                # if "NADI" in self.source:   
                #     pdb.set_trace()
                df = df.query(self.filter_str)
            sents = df[key].values.tolist()
            return sents
        else:
            raise NotImplementedError(f"File type not supported: {ftype}")


class FormatBitext(FormatData):
    """
    """
    def __init__(self, config: dict):
        super().__init__(config)
        # fields needed: in_file_type, src_txt_key, tgt_txt_key, lang_type, src_lang, tgt_lang, source
        self.tgt_txt_key = config['tgt_txt_key']
        self.tgt_lang = self.convert_lang(config['tgt_lang'], self.lang_type)
        self.src_txt_file, self.tgt_txt_file = self.in_files
        self.lang_pair = ">".join([config['src_lang'], config['tgt_lang']])
        self.df_limit = BILIMIT

    def convert_lang(self, lang_str, lang_type):
            """
            We want the lang to actually be readable (not a code)
            """
            code = super().convert_lang(lang_str, lang_type)
            return self.country2dial[code]
    
    def format(self):
        """
        Bitext col's: 
        prompt,completion,source,target_language,source_language,language
        """
        src_txts = self.file2sents(self.src_txt_file, self.src_txt_key, self.ftype)
        tgt_txts = self.file2sents(self.tgt_txt_file, self.tgt_txt_key, self.ftype)
        assert len(src_txts) == len(tgt_txts), f"Bitext mismatched! {len(src_txts)} =/= {len(tgt_txts)}"
        df_dict = {
            "prompt": [MT_PROMPT_TEMP.format(src_lang=self.src_lang, tgt_lang=self.tgt_lang, src_txt=st) \
                       for st in src_txts],
            "completion": tgt_txts,
            "source": [self.source] * len(src_txts),
            "target_language": [self.tgt_lang] * len(src_txts),
            "source_language": [self.src_lang] * len(src_txts),
            "language": [self.tgt_lang] * len(src_txts)
        }
        df = pd.DataFrame(df_dict)
        out_dir = os.path.split(self.out_csv)[0]
        if not os.path.exists(out_dir):
            os.makedirs(out_dir)
        # print("!!!!!!!!!!!!!!!!!!!!!!", self.in_files)
        df = limit_df(df, limit=self.df_limit)
        df.to_csv(self.out_csv)
        print("Written", self.out_csv)
        

class FormatMonotext(FormatData):
    """
    """
    def __init__(self, config: dict):
        super().__init__(config)
        # config fields needed: in_file_type, txt_key, lang, lang_type, source
        self.txt_key = self.src_txt_key
        self.prompt_json = config['prompt_json']
        self.country_code = self.src_lang # self.convert_lang(config['src_lang'], self.lang_type)
        self.lang = self.country2dial[self.country_code]
        self.in_file = self.in_files[0]
        self.lang_pair = config['src_lang']
        self.df_limit = MONOLIMIT

        self.macrogenre = GENRE2MACROGENRE[self.genre]

    def format(self):
        """
        Monotext columns:
        prompt,language,source
        """
        # Open JSON
        with open(self.prompt_json, 'r') as f:
            prompt_dict = json.load(f)
        # dict format: dict[country][genre] = [prompt_temp_list]
        temps = prompt_dict[self.country_code][GENRE2MACROGENRE[self.genre]]
        # Now we construct the prompts
        sents = self.file2sents(self.in_file, self.txt_key, self.ftype)
        prompts = [] 
        for sent in sents:
            chosen_temp = random.choice(temps)
            try:
                prompts.append(chosen_temp.format(sent))
            except:
                pdb.set_trace()
        assert len(prompts) == len(sents)
        # Now create dict and DataFrame
        df_dict = {
            "prompt": prompts,
            "language": [self.lang] * len(prompts),
            "source": [self.source] * len(prompts)
        }
        df = pd.DataFrame(df_dict)
        out_dir = os.path.split(self.out_csv)[0]
        if not os.path.exists(out_dir):
            os.makedirs(out_dir)
        df = limit_df(df, limit=self.df_limit)
        df.to_csv(self.out_csv)
        print("Written", self.out_csv)
    

if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument("--config-json", type=str)
    parser.add_argument("--in-files", type=str, nargs='+')
    parser.add_argument("--langs", type=str, nargs='+')
    parser.add_argument("--text-keys", type=str, nargs='+')
    parser.add_argument("--xtext", type=str, choices=["mono", "bi"])
    parser.add_argument("--file-type", type=str, choices=["csv", "tsv", "txt"])
    parser.add_argument("--lang-type", type=str, choices=["city", "country", "iso"])
    parser.add_argument("--data-src", type=str)
    parser.add_argument("--genre", type=str, choices=["btec", "wiki", "tweets", "music"])
    parser.add_argument("--mono-prompt-json", type=str, default="prompt_templates/json/dial8.json")
    parser.add_argument("--filter-str", type=str, default="split == 'corpus-6-test-corpus-26-dev'")
    parser.add_argument("--out-csv", type=str, default="out.csv")

    args = parser.parse_args()

    if args.config_json:
        with open(args.config_json, 'r') as f:
            config = json.load(f)
        xtext = config['xtext']
        in_files = config["in_files"]
    else:
        xtext = args.xtext
        in_files = args.in_files
        config = {
            "in_files": args.in_files,
            "ftype": args.file_type,
            "src_txt_key": args.text_keys[0],
            "lang_type": args.lang_type,
            "src_lang": args.langs[0],
            "source": args.data_src,
            "genre": args.genre,
            "prompt_json": args.mono_prompt_json,
            "filter_str": args.filter_str,
            "out_csv": args.out_csv
        }
        if xtext == "bi":
            assert len(in_files) == len(args.langs) == len(args.text_keys) == 2
            config["tgt_txt_key"] = args.text_keys[1]
            config["tgt_lang"] = args.langs[1]
        elif xtext == "mono":
            assert len(args.in_files) == len(args.langs) == len(args.text_keys) == 1
    
    for key in ["in_files", "ftype", "src_txt_key", "lang_type", "src_lang", \
                "source", "genre", "prompt_json", "filter_str", "out_csv"]:
        assert key in config, f"Config missing key {key}"


    if xtext == "bi":
        assert "tgt_txt_key" in config and "tgt_lang" in config, "Missing tgt info"
        formater = FormatBitext(config=config)
        formater.format()
    elif xtext == "mono":
        formater = FormatMonotext(config=config)
        formater.format()
