"""
This is the script to create the data in the correct format for
AL-QASIDA, using the raw data from various sources:
    1. FLORES-200
    2. MADAR-26
    3. MADAR-Twitter collection from NADI 2023
    4. HABIBI song lyrics 
"""

from format_xtext import *
from src2info import *
import argparse

# Variables: language, tgt_lang (bi/mono), data_src

# COUNTRIES = ["egy", "jor", "syr", "dza"]
COUNTRIES = ['dza', 'egy', 'kwt', 'mar', 'pse', 'sau', 'sdn', 'syr']

OUT_CSV_TEMPLATE = "data/{xtext}/{genre}/{source}/{langpair}.csv"

def make_out_csv(
        xtext: str, 
        genre: str, 
        source: str, 
        langpair: str, 
        reverse: bool=False
    ) -> str:
    if reverse:  # Need reverse language direction
        langpair = "-".join(langpair.split("-")[::-1]) 
    return  OUT_CSV_TEMPLATE.format(
        xtext=xtext, genre=genre, source=source, langpair=langpair
    ).replace(' ', '_')

def main(json_path):
    for country in COUNTRIES:
        for tgt in ['eng', 'msa', None]:
            if tgt:
                xtext = "bi"
                prompt_json = ""
            else:
                xtext = "mono"
                prompt_json = json_path 
            for source in SRC2INFO: 
                if tgt and not SRC2INFO[source]["bitext"]:
                    continue 
                # (I) PREP ================================================
                # (1) in_files
                if country not in SRC2INFO[source]["countries"]:
                    continue 
                lang = SRC2INFO[source]["from_code"](country)
                extended_lang = SRC2INFO[source]["to_fn_version"](lang)
                in_files = [SRC2INFO[source]["temp"](extended_lang)]
                
                files_exist = True 
                for in_file in in_files:
                    if not os.path.exists(in_file):
                        files_exist = False
                        break
                if not files_exist:
                    print(f"Skipping {source} with {country}, files not found")
                    continue

                langs = [country]
                src_lang = lang
                if tgt:
                    tgt_lang = SRC2INFO[source]["from_code"](tgt)
                    extended_tgt_lang = SRC2INFO[source]["to_fn_version"](
                        tgt_lang
                    )
                    in_files.append(
                        SRC2INFO[source]["temp"](extended_tgt_lang)
                    )
                    langs = [country, tgt]
                # (2) ftype
                if in_files[0].endswith(".tsv"):
                    ftype = "tsv"
                elif in_files[0].endswith(".csv"):
                    ftype = "csv"
                else: # Assuming txt as default
                    ftype = "txt"
                # (3) txt_key's
                src_txt_key = SRC2INFO[source]["txt_keys"][0]
                if tgt:
                    tgt_txt_key = SRC2INFO[source]["txt_keys"][1]
                # (4) lang_type
                lang_type = SRC2INFO[source]["lang_type"]
                # (5) source_name
                source_name = SRC2INFO[source]["name"]
                # (6) genre
                genre = SRC2INFO[source]["genre"]
                # (7) filter_str 
                filter_str = SRC2INFO[source]["c2filter_str"][country]
                # (8) out_csv
                langpair = "-".join(langs)
                out_csv = make_out_csv( # komya
                    xtext=xtext, 
                    genre=genre, 
                    source=source, 
                    langpair=langpair, 
                    reverse=False
                )
                if os.path.exists(out_csv):
                    print(f"WARNING: file {out_csv} exists (skipping)", flush=True)
                    continue
                
                # (II) EXECUTION ==========================================
                # Now for config_dict
                config_dict = {
                    "xtext": xtext,
                    "in_files": in_files,
                    "ftype": ftype,
                    "src_txt_key": src_txt_key,
                    "src_lang": src_lang,
                    "lang_type": lang_type,
                    "source": source_name,
                    "genre": genre,
                    "prompt_json": prompt_json,
                    "filter_str": filter_str,
                    "out_csv": out_csv
                }
                if tgt:
                    config_dict["prompt_json"] = ""
                    config_dict["tgt_txt_key"] = tgt_txt_key 
                    config_dict["tgt_lang"] = tgt_lang 
                    formater = FormatBitext(config=config_dict)
                    print("Formatting for {}".format(formater))
                    formater.format()
                    # Now need to do reverse side 
                    reverse_out_csv = make_out_csv(
                        xtext=xtext, 
                        genre=genre, 
                        source=source, 
                        langpair=langpair, 
                        reverse=True
                    )
                    reverse_config_dict = {
                        "xtext": xtext,
                        "in_files": in_files[::-1],
                        "ftype": ftype,
                        "src_txt_key": tgt_txt_key,
                        "tgt_txt_key": src_txt_key,
                        "src_lang": tgt_lang,
                        "tgt_lang": src_lang,
                        "lang_type": lang_type,
                        "source": source_name,
                        "genre": genre,
                        "prompt_json": "",
                        "filter_str": filter_str,
                        "out_csv": reverse_out_csv
                    }
                    reverse_formater = FormatBitext(
                        config=reverse_config_dict
                    )
                    reverse_formater.format()
                else:
                    formater = FormatMonotext(config=config_dict)
                    print("Formatting for {}".format(formater))
                    formater.format()

if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--json-path", 
        type=str, 
        default="prompt_templates/json/dial8.json"
    )

    args = parser.parse_args()

    main(args.json_path)


