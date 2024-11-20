import re

def parseprompt(promptfile, poem_main_id, poem_a_id, poem_b_id, promptnum=1, fields={}):
    
    promptscheme = open(promptfile)
    
    poem_main = open("corpus/"+poem_main_id+".txt", "r").read()
    poem_a = open("corpus/"+poem_a_id+".txt", "r").read()
    poem_b = open("corpus/"+poem_b_id+".txt", "r").read()
    
    start = False
    prompt = ""
    
    for line in promptscheme:
        
        if line.startswith("<-- StartOfPrompt"+str(promptnum)+" -->"):
            start = True
            continue
            
        if line.startswith("<-- EndOfPrompt"+str(promptnum)+" -->"):
            break
            
        if start:
            
            if line.startswith('{{POEM_MAIN_TEXT}}'):
                prompt += poem_main
                continue
                
            if line.startswith('{{POEM_A_TEXT}}'):
                prompt += poem_a
                continue
                
            if line.startswith('{{POEM_B_TEXT}}'):
                prompt += poem_b
                continue
                
            prompt += line
    
    for keyval in fields.keys():
        
        prompt = re.sub(re.escape(keyval), fields[keyval], prompt, flags=re.DOTALL)
        
    return prompt