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
    
parser = argparse.ArgumentParser(description="Takes a dataset, a prompt form prompts/, a modelname, runs the experiment and stores results")

parser.add_argument('-d', '--dataset', type=str, default="testset", choices=["testset","full","dummy"], help="""Dataset. Choose one of testset/full/dummy
                                                                            testset: 100 Triples
                                                                            full: complete dataset
                                                                            dummy: three triples (for code testing)""")

parser.add_argument('-m', '--model', type=str, choices=['gpt-4o', 'gpt-4o-mini', 'o1-preview','CohereForAI/aya-expanse-32b','VAGOsolutions/Llama-3-SauerkrautLM-70b-Instruct','meta-llama/Llama-3.1-70B-Instruct','claude-3-5-sonnet-20241022', 'claude-3-5-haiku-20241022', 'claude-3-opus-20240229'], help="Name of LLM model")

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
    data = pd.read_csv("dummy.tsv", sep="\t")
    
if dataset == "testset":
    data = pd.read_csv("testset.tsv", sep="\t")
    
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
            
        repfields["{ANSWER_"+str(i)+"}"] = answer
        answer_logger.append(answer)
        
        i+=1

        
    result.append([
        row["base_ID"],row["left_ID"], row["right_ID"], row["triple_ID"], answer, answer_logger
        ])
    
    
# save results to outfile

result = pd.DataFrame(result)
result.columns = ["base_ID","left_ID","right_ID","triple_ID", "answer", "answer_log"]
result["promptscheme"] = promptfile
result["model"] = model
result["flieds"] = repfields
result["dataset"] = dataset
result["sysprompt"] = sysprompt

result.to_csv(outfile, sep="\t")
    
    
    