"""
This is the primary file to run the AL-QASIDA evaluation.
See README for example commands.
"""

import numpy as np
import pandas as pd 
import torch
from tqdm import tqdm 
from transformers import(
    AutoTokenizer, 
    AutoModelForCausalLM, 
    AutoModelForSequenceClassification,
    pipeline 
) 

import fasttext
from huggingface_hub import hf_hub_download
from sacrebleu.metrics import BLEU, CHRF

import os, glob, argparse
import pickle as pkl 
import pdb  
from typing import List

from tokens import HF_TOKEN
import jais 
from data_organizers import InDataOrganizer, OutDataOrganizer
from maps import (
    DIALECTS, 
    COUNTRY2DIALECT, 
    DIALECT2COUNTRY, 
    COUNTRY2MACRO_DIALECT, 
    MICROLANGUAGE_MAP, 
    TASK2DIALECTS
)

SUPPORTED_MODELS = ["jais", "acegpt", "silma", "llama", "llama-base"]
MOD2HF_NAME = {
    "jais": "core42/jais-13b",
    "silma": "silma-ai/SILMA-9B-Instruct-v1.0",
    "acegpt": "FreedomIntelligence/AceGPT-7B-chat",
    "llama": "meta-llama/Meta-Llama-3.1-8B-Instruct",
    "llama-base": "meta-llama/Meta-Llama-3.1-8B", 
    "aldi": "AMR-KELEG/Sentence-ALDi", 
    "nadi": "AMR-KELEG/NADI2024-baseline",
} 
TASK2ABBREV = {
    "monolingual": "mono",
    "crosslingual": "xling",
    "mt": "bi"
}
NSHOT2JSON = {
    5: "5shot_prefixes.json"
}
TASK2NSHOT_SUFFIX = {
    "monolingual": "\nA: "
}
GENERATION_LIMIT = 128
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

fT_model_path = hf_hub_download(
    repo_id="facebook/fasttext-language-identification", 
    filename="model.bin"
)
print(f"fastText model stored in {fT_model_path}")

def _sacre_bleu_score(labels, generations, scorer):
    hyps, refs = [], [[]]
    for l, g in zip(labels, generations):
        l, g = l.strip(), g.strip()
        refs[0].append(l)
        hyps.append(g)

    if hyps == [] and refs == [[]]:
        return None
    return scorer.corpus_score(hyps, refs)

def spbleu_corpus_score(labels: List[str], generations: List[str]):
    """SentencePiece BLEU score, as implemented in sacrebleu."""
    spbleu = BLEU(tokenize="flores200")
    result = _sacre_bleu_score(labels, generations, spbleu)
    return {"SpBLEU_corpus_score": result.score / 100 if result else None}

def chrf_corpus_score(labels: List[str], generations: List[str]):
    """chrF2 score, as implemented in sacrebleu."""
    chrf = CHRF()
    result = _sacre_bleu_score(labels, generations, chrf)
    return {"ChrF_corpus_score": result.score / 100 if result else None}

class BaseEvaluator():
    def __init__(self, llm_type, dialect, target_lang='ara'):
        self.llm_type = llm_type 
        self.llm, self.llm_tokenizer = self.llm_type2llm(llm_type)
        self.run_llm = self.get_run_llm(llm_type)
        self.aldi_model, self.aldi_tokenizer = self.load_aldi() 
        self.nadi_model, self.nadi_tokenizer = self.load_nadi()
        self.lid_model = self.load_lid() 
        self.dialect = dialect
        self.target_lang = target_lang 
    
    def llm_type2llm(self, llm_type): 
        if llm_type == "jais":
            return self.load_jais() 
        elif llm_type in ["silma", "acegpt"]: 
            return self.load_silma_acegpt(llm_type) 
        elif llm_type.startswith("llama"):
            return self.load_llama() 
        else:
            raise NotImplementedError(
                f"Only {SUPPORTED_MODELS} models supported, not {llm_type}"
            )
    
    def get_run_llm(self, llm_type):
        if llm_type == "jais":
            return self.run_jais 
        elif llm_type in ["silma", "acegpt", "llama", "llama-base"]: 
            return self.run_hf_pipeline
        else:
            raise NotImplementedError(
                f"Must be Llama or Jais model: {llm_type}"
            )
    
    def dialect2index(self, dialect):
        return DIALECTS.index(COUNTRY2DIALECT[dialect])
    
    def clean_text(self, text):
        return text.replace("\n", " ") # Add other cleaning steps here

    def get_macro_prob(self, probabilities, dialect):
        return sum([
            probabilities[i] for i in range(
                len(DIALECTS)
            ) if COUNTRY2MACRO_DIALECT[
                DIALECT2COUNTRY[DIALECTS[i]]
            ] == COUNTRY2MACRO_DIALECT[dialect]
        ])
    
    def process_lid_label(self, model_out):
        label = model_out[0][0]
        assert len(label) == 9 + 3 + 5
        assert label[0] == label[1] == label[7] == label[8] == label[-5] ==\
                '_'
        return label[9:-5]
    
    def load_aldi_nadi(self, option="aldi"): 
        # -> (model, tokenzier)
        model_name = MOD2HF_NAME[option]
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForSequenceClassification.from_pretrained(model_name)
        return model, tokenizer
    
    def load_aldi(self): 
        print("Loading ALDi model...", flush=True)
        return self.load_aldi_nadi(option="aldi")
    
    def load_nadi(self):
        print("Loading NADI model...", flush=True)
        return self.load_aldi_nadi(option="nadi")
    
    def load_lid(self):
        # -> model 
        print("Loading fastText model...", flush=True)
        model = fasttext.load_model(fT_model_path) 
        return model
    
    def run_aldi(self, text): 
        # text -> dialectness 
        inputs = self.aldi_tokenizer(text, return_tensors="pt")
        outputs = self.aldi_model(**inputs)
        logits = outputs.logits
        return min(max(0, logits[0][0].item()), 1)  
    
    def run_nadi(self, text, dialect="egy"): 
        # (text, dialect) -> (prob, macro_prob) 
        logits = self.nadi_model(
            **self.nadi_tokenizer(text, return_tensors="pt")
        ).logits 
        probabilities = torch.softmax(logits, dim=1).flatten().tolist()
        
        # Calculate prob 
        dialect_idx = self.dialect2index(dialect)
        prob = probabilities[dialect_idx] 

        # Calculate macro_prob 
        macro_prob = self.get_macro_prob(probabilities, dialect)

        return prob, macro_prob 
    
    def run_lid(self, text): 
        # text -> lang_id (bool-like) 
        label = self.process_lid_label(self.lid_model.predict(text)) 
        if label in MICROLANGUAGE_MAP[self.target_lang]:
            return 1 
        return 0  
    
    def load_jais(self):
        print("Loading JAIS model...", flush=True)
        model_path = MOD2HF_NAME[self.llm_type]
        tokenizer = AutoTokenizer.from_pretrained(
            model_path, 
            use_auth_token=HF_TOKEN,
            device=DEVICE
        )
        model = AutoModelForCausalLM.from_pretrained(
            model_path, 
            device_map={"": DEVICE}, 
            trust_remote_code=True, 
            use_auth_token=HF_TOKEN,
        ).to(torch.device(DEVICE))
        return model, tokenizer
    
    def load_silma_acegpt(self, option="silma"): 
        print(f"Loading {option.upper()} model...", flush=True)
        model_id = MOD2HF_NAME[self.llm_type]
        pipe = pipeline(
            "text-generation", 
            model=model_id, 
            model_kwargs={"torch_dtype": torch.bfloat16}, 
            device=DEVICE, # HACK 
        ) 
        return pipe, None 
    
    def load_llama(self): 
        print(f"Loading Llama model...", flush=True)
        model_id = MOD2HF_NAME[self.llm_type]
        pipe = pipeline(
            "text-generation", 
            model=model_id, 
            model_kwargs={"torch_dtype": torch.bfloat16}, 
            device=DEVICE, 
            token=HF_TOKEN
        ) 
        return pipe, None  

    def run_jais(self, prompts): 
        assert self.llm_type == "jais"
        outs = [] 
        for prompt in tqdm(prompts):
            outs.append(
                jais.get_response(
                    prompt,
                    tokenizer=self.llm_tokenizer,
                    model=self.llm
                ) 
            )
        return outs 
    
    def run_hf_pipeline(self, prompts, stride=50):
        subset_points = [i for i in range(len(prompts)) if i % stride == 0]
        outs = [] 
        for subset_point in tqdm(subset_points):
            subset_prompts = prompts[subset_point: subset_point + stride]
            outs += [
                item[0]['generated_text'] for item in self.llm(
                    subset_prompts, 
                    return_full_text=False, 
                    max_new_tokens=GENERATION_LIMIT,
                    pad_token_id=self.llm.tokenizer.eos_token_id
                )
            ]
        # clean_outs = [] 
        # for out, prompt in zip(outs, prompts):
        #     clean_out = out.strip() 
        #     if clean_out.startswith(prompt): 
        #         clean_out = clean_out[len(prompt):].strip() 
        #     clean_outs.append(clean_out) 
        return outs
    
    def adi2_eval(self, prompts):
        all_scores = {
            "prob": [], 
            "dialectness": [], 
            "score": [], 
            "macro_score": []
        } 
        print(
            f"Running LLM for {self.dialect} {self.llm_type}...", 
            flush=True
        )
        outputs = self.run_llm(
            [self.clean_text(prompt) for prompt in prompts]
        )
        print(
            f"Getting scores for {self.dialect} {self.llm_type}...", 
            flush=True
        )
        for output in tqdm(outputs): 
            clean_output = self.clean_text(output)
            # Check if output is in right language 
            lang_id = self.run_lid(clean_output) if clean_output else 0
            if not lang_id or self.target_lang != "ara" or self.dialect == "msa":
                for key in all_scores:
                    all_scores[key].append(0.)
                continue 
            prob, macro_prob = self.run_nadi(clean_output, self.dialect)
            dialectness = self.run_aldi(clean_output) 
            score = prob * dialectness 
            macro_score = macro_prob * dialectness 
            # Collect scores
            all_scores["prob"].append(prob)
            all_scores["dialectness"].append(dialectness) 
            all_scores["score"].append(score) 
            all_scores["macro_score"].append(macro_score) 
        # Collect averages 
        mean_score_dict = {
            key: np.mean(all_scores[key]) for key in all_scores
        }
        return mean_score_dict, outputs


class MTEvaluator(BaseEvaluator):
    def __init__(self, llm_type, mt_direction="eng-egy"):
        dialect, target_lang = self.retrieve_dialect_lang_from_direction(
            mt_direction
        )
        self.mt_direction = mt_direction 
        super().__init__(
            llm_type=llm_type, 
            dialect=dialect, 
            target_lang=target_lang
        ) 
    
    def retrieve_dialect_lang_from_direction(self, mt_direction):
        dialect = mt_direction.split('-')[-1]
        target_lang = "eng" if dialect == "eng" else "ara"
        return dialect, target_lang 
    
    def __call__(self, prompts, refs):
        mean_score_dict, hyps = self.adi2_eval(prompts) 
        # Want "SpBLEU_corpus_score" and "ChrF_corpus_score"
        mean_score_dict.update(
            spbleu_corpus_score(labels=refs, generations=hyps)
        )
        mean_score_dict.update(
            chrf_corpus_score(labels=refs, generations=hyps)
        )
        return mean_score_dict, hyps 


class LingualEvaluator(BaseEvaluator): 
    def __call__(self, prompts):
        return self.adi2_eval(prompts)


def run_evaluation(
        data_dir, # data/mono or data/xling
        out_pkl,
        out_dir,
        task="monolingual",
        test_bool=False,
        llms=SUPPORTED_MODELS,
        dialects=["dza", "mar", "egy", "sdn", "pse", "syr", "sau", "kwt"],
        nshot=0,
    ):
    in_data_organizer = InDataOrganizer(data_dir, task=task, test=test_bool) 
    prompt_organization = in_data_organizer.organize_prompts() 
    if task == "mt":
        ref_organization = in_data_organizer.organize_prompts(mt_refs=True) 
    output_organization = {genre: {} for genre in prompt_organization}
    out_prompt_organization = {genre: {} for genre in prompt_organization}
    total_evals = len(llms) * sum(
        [len(prompt_organization[genre]) for genre in prompt_organization]
    )
    # Set up nshots
    lang2prefix = {}
    if nshot:
        with open(NSHOT2JSON[nshot], 'r') as f:
            task2lang2prefix = json.load(f) 
        if task in task2lang2prefix:
            lang2prefix = task2lang2prefix[task]
        else:
            print(
                f"WARNING: nshot set to {nshot}, "\
                f"but no shot strings available for {task},"\
                f"Defaulting to 0-shot"
            )
    # Actual loop
    count_idx = 0
    for llm in llms: # In output must be dir with task 
        if llm not in out_pkl:
            out_pkl += "_" + llm
        for genre in prompt_organization: 
            for dialect in prompt_organization[genre]:
                count_idx += 1
                print("#####" * 10) 
                print(f"Running evaluation for {dialect} {genre} {llm}" + \
                    f" ({count_idx}/{total_evals})")
                print("#####" * 10, flush=True)
                prompts = prompt_organization[genre][dialect] 
                if dialect in lang2prefix: # Add n shots
                    prefix = lang2prefix[lang]
                    suffix = ""
                    if task in TASK2NSHOT_SUFFIX:
                        suffix = TASK2NSHOT_SUFFIX[task]
                    print(f"Using prefix = {prefix}")
                    print(f"And using suffix = {suffix}", flush=True)
                    prompts = [prefix + prompt + suffix for prompt in prompts]
                # Run eval
                if task == "mt":
                    evaluator = MTEvaluator(llm_type=llm, mt_direction=dialect)
                    refs = ref_organization[genre][dialect]
                    mean_score_dict, outs = evaluator(prompts, refs)
                else:
                    evaluator = LingualEvaluator(llm_type=llm, dialect=dialect)
                    mean_score_dict, outs = evaluator(prompts) 
                output_organization[genre][dialect] = mean_score_dict 
                out_prompt_organization[genre][dialect] = outs 

                with open(out_pkl + "_scores.pkl", 'wb') as f:
                    pkl.dump(output_organization, f)
                with open(out_pkl + "_completions.pkl", 'wb') as f:
                    pkl.dump(out_prompt_organization, f)
                
                print(); del evaluator 
        
        # Out data organization 
        out_data_organizer = OutDataOrganizer(
            data_dict=output_organization, 
            llm=llm, 
            task=task, 
            reports_dir=out_dir
        )
        out_data_organizer.organize_data()

if __name__ == "__main__":

    parser = argparse.ArgumentParser() 
    parser.add_argument("--data-dir", required=True, type=str)
    parser.add_argument("--test", action="store_true")
    parser.add_argument("--cpu", action="store_true") 
    parser.add_argument("--out-dir", default="../llm_outputs", type=str)
    parser.add_argument("--task", default="monolingual", type=str)
    parser.add_argument(
        "--llm", 
        default=None, 
        choices=SUPPORTED_MODELS
    )
    parser.add_argument(
        "--nshot", 
        type=int,
        default=0, 
        choices=[0,5]
    )
    args = parser.parse_args() 

    if not (args.cpu or torch.cuda.is_available()):
        raise RuntimeError("Need GPU")
    
    out_pkl_dir = os.path.join(args.out_dir, "pkls")
    if not os.path.exists(out_pkl_dir):
        os.makedirs(out_pkl_dir) 
    
    if args.llm:
        llm_list = [args.llm]
        out_pkl_fn = os.path.join(out_pkl_dir, f"{args.task}_{args.llm}")
    else:
        llm_list = ["jais", "acegpt", "silma", "llama", "llama-base"]
        out_pkl_fn = os.path.join(out_pkl_dir, args.task)
    
    task_abbrev = TASK2ABBREV[args.task]

    run_evaluation(
        data_dir=os.path.join(args.data_dir, task_abbrev), 
        out_pkl=out_pkl_fn,
        out_dir=args.out_dir,
        task=args.task,
        llms=llm_list,
        dialects=TASK2DIALECTS[args.task],
        test_bool=args.test,
        nshot=args.nshot,
    )



