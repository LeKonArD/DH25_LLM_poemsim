# DH25_LLM_poemsim
## Configuration
Create a file .env and configure it like this: <br>
OPENAI_KEY=InsertYourKeyHere <br>
ANTHROPIC_KEY=InsertYourKeyHere <br>

## Requirements
python >= 3.12 <br>
openai==1.54.5 <br>
pandas==2.2.2 <br>
python-dotenv==1.0.1 <br>
anthropic==0.39.0 <br>
numpy==1.26.4 <br>
scikit-learn==1.5.0 <br>
scipy==1.13.1 <br>

## Usage

The script start_run.py takes a datasets and and a prompt scheme from prompts/ formats
everything and stores responses.<br>

'-d', '--dataset', type=str, default="testset", choices=["testset","full","dummy"], help="""Dataset. Choose one of testset/full/dummy
                                                                            testset: 100 Triples
                                                                            full: complete dataset
                                                                            dummy: three triples (for code testing)"""<br>

'-m', '--model', type=str, choices=['gpt-4o', 'gpt-4o-mini', 'o1-preview','CohereForAI/aya-expanse-32b','VAGOsolutions/Llama-3-SauerkrautLM-70b-Instruct','meta-llama/Llama-3.1-70B-Instruct'], help="Name of LLM model"<br>

'-p', '--promptfile', type=str, help='location of the prompt.txt'<br>

'-rf', '--replacefields',  action=StoreDictKeyPair, nargs="+", metavar="KEY=VAL", default={}, help="""Dictionary to replace a keyphrase with a string: Example: {DIMENSION}=Style"""<br>

'-s', '--sysprompt', type=str, help='System prompt', default="default"<br>
'-o', '--outfile', type=str, help='name/path to write results'<br>

## Exmaple of a prompt scheme
<-- StartOfPromptFile --><br>
<-- StartOfPrompt1 --><br>
Your Task is to judge the similarity of poems. Consider the following three poems (POEM_MAIN, POEM_A, POEM_B) and judge whether POEM_MAIN is more similar to POEM_A than POEM_B or more similar to POEM_B than POEM_A or equally (dis)similar to both POEM_A and POEM_B:<br>

POEM_MAIN:<br>
{{POEM_MAIN_TEXT}}<br>
<br>
POEM_A:<br>
{{POEM_A_TEXT}}<br>
<br>
POEM_B:<br>
{{POEM_B_TEXT}}<br>

In your judgment, focus on {DIMENSION}. Do not consider any other text dimensions.<br>

Format your answer like this:<br>
Begin with ‚ANSWER: ‘<br>
If POEM_MAIN is more similar to POEM_A than POEM_B, write ‘POEM_A’.<br>
If POEM_MAIN is more similar to POEM_B than POEM_A, write ‚POEM_B‘.<br>
If POEM_MAIN is equally (dis)similar to both POEM_A and POEM_B, write ‚BOTH‘.<br>
After ‚ANSWER: ‘, do not write anything else than ‘POEM_A’, ‘POEM_B’ or ‚BOTH‘.<br>
<br>
For example, your complete answer could look like this:<br>
<br>
ANSWER: BOTH<br>
<-- EndOfPrompt1 --><br>
<-- EndOfPromptFile --><br>

The poems text is filled in automatically. The Input of {DIMENSION} can be filled by using --replacefields from start_run.py 