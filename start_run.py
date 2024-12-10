import argparse
import os
import sys
import pandas as pd

from pathlib import Path
from utils import parseprompt, get_num_chains
from dotenv.main import load_dotenv

load_dotenv()
## globals

openai_models = ['gpt-4o', 'gpt-4o-mini', 'o1-preview']
anthropic_models = ['claude-3-5-sonnet-20241022', 'claude-3-5-haiku-20241022', 'claude-3-opus-20240229']
hf_models = ['CohereForAI/aya-expanse-32b',
             'meta-llama/Llama-3.1-70B-Instruct',
             'VAGOsolutions/Llama-3-SauerkrautLM-70b-Instruct',
             'openGPT-X/Teuken-7B-instruct-research-v0.4',
             'google/gemma-2-27b-it',
             'microsoft/Phi-3-medium-4k-instruct',
             'mistralai/Mixtral-8x7B-Instruct-v0.1',
             'Qwen/Qwen2-7B-Instruct'
             ]

## parse agruments

class StoreDictKeyPair(argparse.Action):
    
    def __init__(self, option_strings, dest, nargs=None, **kwargs):
        self._nargs = nargs
        super(StoreDictKeyPair, self).__init__(option_strings, dest, nargs=nargs, **kwargs)
        
    def __call__(self, parser, namespace, values, option_string=None):
        my_dict = {}
        print("values: {}".format(values))
        for kv in values:
            k,v = kv.split("=")
            my_dict[k] = v
            
        setattr(namespace, self.dest, my_dict)

def promptOPENAI(coreprompt, sysprompt, model):
    
    
    if sysprompt == "default":
        
        chat_completion = client.chat.completions.create(
        messages=[
                {
                    "role": "user",
                    "content": coreprompt,
                }
            ],
            model=model)


        answer = chat_completion.choices[0].message.content
            
    else:
        
        chat_completion = client.chat.completions.create(
        messages=[
                {
                 "role": "system", 
                 "content": sysprompt,
                },
                {
                    "role": "user",
                    "content": coreprompt,
                },
            ],
            model=model)


        answer = chat_completion.choices[0].message.content
    
    return answer
        
    
def promptANTHROPIC(coreprompt, sysprompt, model):
    
    
    if sysprompt == "default":
        
        message = client.messages.create(
                            model=model,
                            max_tokens=1024,
                            messages=[
                                {"role": "user", "content": coreprompt}
                            ]
                        )
        
        answer = message.content[0].text
            
    else:
        
        message = client.messages.create(
                            model=model,
                            max_tokens=1024,
                            messages=[
                                {"role": "system", "content": sysprompt},
                                {"role": "user", "content": coreprompt}
                            ]
                        )


        answer = message.content[0].text
    
    return answer
  
def promptHF(coreprompt, sysprompt):
    
    if sysprompt == "default":
        
        if model == "CohereForAI/aya-expanse-32b" or model == "VAGOsolutions/Llama-3-SauerkrautLM-70b-Instruct":
        
            sysprompt = "Du bist ein freundlicher und hilfreicher deutscher KI-Assistent."
            
        if model == "meta-llama/Llama-3.1-70B-Instruct":
        
            sysprompt = "You are a helpful assistant."
            
        if model == "openGPT-X/Teuken-7B-instruct-research-v0.4":
        
            sysprompt = "Ein Gespräch zwischen einem Menschen und einem Assistenten mit künstlicher Intelligenz. Der Assistent gibt hilfreiche und höfliche Antworten auf die Fragen des Menschen."   
            
           
            
    if model == 'CohereForAI/aya-expanse-32b':
        
        
        prompt_format = """
        <BOS_TOKEN><|START_OF_TURN_TOKEN|><|USER_TOKEN|>{coreprompt}<|END_OF_TURN_TOKEN|><|START_OF_TURN_TOKEN|><|CHATBOT_TOKEN|>
        """
        inputs = prompt_format.format(coreprompt=coreprompt)
        
    if model == "meta-llama/Llama-3.1-70B-Instruct" or model == "VAGOsolutions/Llama-3-SauerkrautLM-70b-Instruct":
    
        prompt_format = """
        <|begin_of_text|><|start_header_id|>system<|end_header_id|>{sysprompt}<|eot_id|><|start_header_id|>user<|end_header_id|>{coreprompt}<|eot_id|><|start_header_id|>assistant<|end_header_id|>
        """
        
        inputs = prompt_format.format(coreprompt=coreprompt,sysprompt=sysprompt)
        
    if model == "openGPT-X/Teuken-7B-instruct-research-v0.4":
        
        prompt_format = """
        System: {sysprompt}\nUser: {coreprompt}\nAssistant:
        """
        
        inputs = prompt_format.format(coreprompt=coreprompt,sysprompt=sysprompt)
        
    if model == "google/gemma-2-27b-it":
        
        prompt_format = '<bos><start_of_turn>user\n{coreprompt}<end_of_turn>\n'
        inputs = prompt_format.format(coreprompt=coreprompt)
        
    if model == "microsoft/Phi-3-medium-4k-instruct":
        
        prompt_format = '<|user|> {coreprompt} <|end|> <|assistant|>'
        inputs = prompt_format.format(coreprompt=coreprompt)
    
    if model == "mistralai/Mixtral-8x7B-Instruct-v0.1":
        
        prompt_format = '<s> [INST] {coreprompt} [/INST]'
        inputs = prompt_format.format(coreprompt=coreprompt)
        
    if model == "Qwen/Qwen2-7B-Instruct":
        
        prompt_format = '<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n<|im_start|>user\n{coreprompt}<|im_end|>\n'
        inputs = prompt_format.format(coreprompt=coreprompt)
        
        
    input_token = tokenizer.encode_plus(inputs, return_tensors="pt")
    output = HFmodel.generate(input_token["input_ids"].to("cuda"), 
                            attention_mask=input_token["attention_mask"].to("cuda"),
                            max_new_tokens=1024)
    
   
    
    return tokenizer.decode(output[0])
 
parser = argparse.ArgumentParser(description="Takes a dataset, a prompt form prompts/, a modelname, runs the experiment and stores results")

parser.add_argument('-d', '--dataset', type=str, default="testset", choices=["testset","konsens","konsens+","konsensALL","full","dummy"], help="""Dataset. Choose one of testset/full/dummy
                                                                            testset: 100 triples
                                                                            konsens: 90 high quality triples
                                                                            konsens+: 150 more high quality triples
                                                                            konsensALL: konsens and konsens+
                                                                            full: complete dataset
                                                                            dummy: three triples (for code testing)""")

parser.add_argument('-m', '--model', type=str, choices=['gpt-4o', 'gpt-4o-mini', 'o1-preview','CohereForAI/aya-expanse-32b','VAGOsolutions/Llama-3-SauerkrautLM-70b-Instruct','meta-llama/Llama-3.1-70B-Instruct','claude-3-5-sonnet-20241022', 'claude-3-5-haiku-20241022', 'claude-3-opus-20240229','openGPT-X/Teuken-7B-instruct-research-v0.4',
             'google/gemma-2-27b-it',
             'microsoft/Phi-3-medium-4k-instruct',
             'mistralai/Mixtral-8x7B-Instruct-v0.1',
             'Qwen/Qwen2-7B-Instruct'], help="Name of LLM model")

parser.add_argument('-p', '--promptfile', type=str, help='location of the prompt.txt')

parser.add_argument('-rf', '--replacefields',  action=StoreDictKeyPair, nargs="+", metavar="KEY=VAL", default={}, help="""Dictionary to replace a keyphrase with a string: Example: {DIMENSION}=Style""")

parser.add_argument('-s', '--sysprompt', type=str, help='System prompt', default="default")
parser.add_argument('-o', '--outfile', type=str, help='name/path to write results')

args = parser.parse_args()

dataset = args.dataset
model = args.model
promptfile = args.promptfile
outfile = args.outfile
repfields = args.replacefields
sysprompt = args.sysprompt

## check arguments

if os.path.isfile(promptfile) == False:
    
    sys.exit("ERROR: Could not find promptfile at: "+promptfile)

try:
    
    with open(outfile, "w") as f:
        f.write("test")
        
    os.remove(outfile)
    
except:
    
    sys.exit("ERROR: Could not write outfile to: "+outfile)

    

### load dataset

if dataset == "dummy":
    data = pd.read_csv("SW_data.tsv", sep="\t")
    
if dataset == "testset":
    data = pd.read_csv("testset.tsv", sep="\t")
    
if dataset == "konsens":
    data = pd.read_csv("konsens_dataset.tsv", sep="\t")
    
if dataset == "konsens+":
    data = pd.read_csv("konsens+.tsv", sep="\t")
    
if dataset == "konsensALL":
    
    data1 = pd.read_csv("konsens_dataset.tsv", sep="\t")
    data2 = pd.read_csv("konsens+.tsv", sep="\t")
    data = pd.concat([data1, data2], axis=0)
    
if dataset == "full":
    data = pd.read_csv("full_dataset.tsv", sep="\t")
    
### login openAI

if model in openai_models:
    
    import openai
    
    openai_key = os.getenv("OPENAI_KEY")
    client = openai.OpenAI(api_key=openai_key)
    
### login to anthropic

if model in anthropic_models:
    
    import anthropic
    
    anthropic_key = os.getenv("ANTHROPIC_KEY")
    client = anthropic.Anthropic(api_key=anthropic_key)

### setup local prompting ###
   
if model in hf_models:
    
    os.environ["CUDA_VISIBLE_DEVICES"]="0"
    os.environ['TRANSFORMERS_CACHE'] = os.getenv("TRANSFORMERS_CACHE")
    
    from transformers import pipeline, BitsAndBytesConfig, AutoTokenizer, AutoModelForCausalLM
    qc = BitsAndBytesConfig(load_in_4bit=True)
    HFmodel = AutoModelForCausalLM.from_pretrained(model, trust_remote_code=True,
                                                 quantization_config=qc)

    tokenizer = AutoTokenizer.from_pretrained(model, trust_remote_code=True)

    tokenizer.pad_token_id = tokenizer.eos_token_id
    HFmodel.config.pad_token_id = tokenizer.pad_token_id
    
    
### execute prompting

result = []

for index, row in data.iterrows():
    
    chain_num = get_num_chains(promptfile)
    answer_logger = []
    
    i = 1
    
    while i <= chain_num:
        
    
        coreprompt = parseprompt(promptfile=promptfile, 
                                 poem_main_id=row["base_ID"], 
                                 poem_a_id=row["left_ID"], 
                                 poem_b_id=row["right_ID"],
                                 promptnum=i,
                                 fields=repfields)
 
        if model in openai_models:

            
            answer = promptOPENAI(coreprompt, sysprompt, model)
            
        if model in anthropic_models:
            
            answer = promptANTHROPIC(coreprompt, sysprompt, model)
            
        if model in hf_models:
            
            answer = promptHF(coreprompt, sysprompt)
           
            
        repfields["{ANSWER_"+str(i)+"}"] = answer
        answer_logger.append(answer)
        
        i+=1

        
    result.append([row["base_ID"],row["left_ID"], row["right_ID"], row["triple_ID"], answer, answer_logger])
    
    
    # save results to outfile (each prompt to aviod loss in case of server overload)

    res = pd.DataFrame(result)
    res.columns = ["base_ID","left_ID","right_ID","triple_ID", "answer", "answer_log"]
    res["promptscheme"] = promptfile
    res["model"] = model
    res["flieds"] = repfields
    res["dataset"] = dataset
    res["sysprompt"] = sysprompt

    res.to_csv(outfile, sep="\t")
    
    
    