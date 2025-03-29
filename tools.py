import pandas as pd 
from pydantic import BaseModel, Field
from typing import List, Optional, Annotated, Union, Tuple
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
import os 
from dotenv import find_dotenv, load_dotenv
import random 
import json 
from sklearn.metrics import cohen_kappa_score
from collections import Counter
from sklearn.metrics.pairwise import cosine_similarity
from transformers import BertModel, BertTokenizer
import torch
import numpy as np
import requests
from concurrent.futures import ThreadPoolExecutor



## Specify the model name 
## Initalize a tokenizer that converts words in a sentence to token ids 
## Use the BERT model to read in token ids to call specific arrays 
## Using torch.no_grad() we do a forward pass where we take the vector of inputs for each sentence in the model
model_name = 'bert-base-uncased'
bert_tokenizer = BertTokenizer.from_pretrained(model_name)
bert_model = BertModel.from_pretrained(model_name)
bert_model.eval()

def generate_json_str(codes = None):
    """Generate example outputs of JSON object given few shot example human-labels
    Args:
        codes (tuple): A tuple of COUNTS (sum of digital design codes), SHORT DESCRIPTIONS of each design, and REASON for why design satisfies inclusion criteria
    Raises:
        KeyError: if the length of descriptions or reasons does not match the COUNT
    """
    if not codes:
        return "N/A"
    
    ### Move the generating example actions after the formating 
    count, short_descripts, reason = codes 

    if count == 1: 
        if (len([short_descripts]) != count) or (len([reason]) != count):
            raise KeyError("Descriptions or reasoning processes documented mismatched with codes.")
    if count > 1: 
        if (len(short_descripts) != count) or (len(reason) != count):
            raise KeyError("Descriptions or reasoning processes documented mismatched with codes.")
    
    
    actions = []
    for i in range(count):
        action = {
            "id": f"A{i+1}",
            "summary": short_descripts[i],
            "reasons": reason[i]
        }
        actions.append(action)
    
    output = {"actions": actions, "count": count}
    return json.dumps(output, indent=4)

# Outputs a nicely constructed prompt given this information 

def construct_prompt(coding_instructions: str, 
                     few_shot: bool,
                     continuation_text: str,
                     few_shot_num: Optional[int] = 4,
                     few_shot_texts: Optional[List[str]] = None, 
                     few_shot_codes: Optional[List[int]] = None
                     ) -> str:
    """Construct prompts for the dataframe, randomizes examples, randomizes the deductive codes to prevent LLM recency bias and 
    focus on internalizing the concept extraction tasks

    Args:
        coding_instructions (str): Manual instructions to type out 
        few_shot (bool): Whether the model is few shot or zero shot 
        few_shot_texts (List[str]): A few text examples 
        few_shot_codes (List[int]): A few deductively coded examples for the corresponding texts
    """
    ## Initial instruction 
    prompt = coding_instructions + "\n\n"

    prompt_out = ChatPromptTemplate.from_template(prompt + "{exemplar_icl}" + "{continuation_text}" + "\n\nAnswer:\n")

    if few_shot == True: 
        ## Permute examples to limit recency bias (Zhao et al. 2021)
        examples = list(zip(few_shot_texts, few_shot_codes))
        random.shuffle(examples)
        few_shot_texts, few_shot_codes = zip(*examples)
        
        few_shot_examples = ""
        for i in range(min(few_shot_num, len(few_shot_codes))): 
            json_out = generate_json_str(few_shot_codes[i])
            few_shot_examples += "\n###\nExamples:" + few_shot_texts[i] + "\nAnswer:\n" + json_out

        ## Provide contextual calibration (Zhao et al. 2021)
        few_shot_examples += "\n###\nExamples: " + "N/A [MASK] [MASK]" + "\nAnswer:\n" + "N/A" + "\n\n"
    if few_shot == False: 
        few_shot_examples = " "
        
    formatted_prompt = prompt_out.format(exemplar_icl=few_shot_examples, continuation_text=continuation_text)
    return formatted_prompt

def parse_json_string(response: str):
    """Decodes LLM output (str) into JSON object"""
    try: 
        response = response.strip('```').strip('json')
        json_output = json.loads(response)
        return json_output 
    except json.JSONDecodeError:
        print('Failed to decode JSON object.')
        return None 
    
def parse_val_from_json_string(json_string, val):
    """Extract output from JSON object such as "count", "actions" """
    lst = []
    try:  
        json_out = parse_json_string(json_string)
        if val == "count" or val == "actions": 
            return json_out[val]
        else: 
            for item in json_out['actions']: 
                lst.append(item[val])       
            return lst
    except Exception as e: 
        print("Error: {e}")
        return "N/A" 

json_string = """{
    "actions": [
        {
            "id": "A1",
            "summary": "TikTok introduced Family Pairing features to help parents set boundaries and limits for their teens' accounts.",
            "reasons": "This action meets I1 as it focuses on user safety and well-being by giving parents more control over their teens' digital habits. It meets I2 as it involves a new feature implementation on the platform. It meets I3 as the feature is already launched and available to users."
        },
        {
            "id": "A2",
            "summary": "Implemented a wind down feature to encourage teens to switch off TikTok after 10pm.",
            "reasons": "This action satisfies I1 as it targets user behaviors relevant to well-being by promoting balanced digital habits. It meets I2 as it involves a new in-app feature being introduced. It meets I3 as the feature has been rolled out to users."
        },
        {
            "id": "A3",
            "summary": "Added over 15 safety, well-being, and privacy features for parents to view or adjust in their teens' accounts.",
            "reasons": "This action aligns with I1 as it enhances user privacy and safety by giving more control to parents. It meets I2 as it involves the addition of multiple new features. It meets I3 as the features are already available for parents to use."
        },
        {
            "id": "A4",
            "summary": "Partnered with Telefónica to explore age assurance methods and support industry-wide dialogue on age assurance.",
            "reasons": "This action satisfies I1 as it involves age-appropriate design considerations and user safety. It meets I2 as it involves a collaboration with another organization to explore age verification methods. It meets I3 as the partnership and dialogue initiatives are ongoing."
        }
    ],
    "count": 4
}"""
jkr = parse_val_from_json_string(json_string=json_string, val="reasons")

for i, val in enumerate(jkr):
    print(i, val)


        
    
def compute_cohen_kappa(evaluation_df, machine_coder, y1, y2): 
    """Passes in a dataframe of three columns with values of machine coders and two humans.
    Args:
        evaluation_df (pandas.DataFrame): pandas.DataFrame name 
        machine_coder (str): name of machine coder column
        y1 (str): name of human coder 1 column
        y2 (str): name of human coder 2 column 
    """
    y1_k = cohen_kappa_score(evaluation_df[machine_coder], evaluation_df[y1])
    y2_k = cohen_kappa_score(evaluation_df[machine_coder], evaluation_df[y2])
    between_humans_k = cohen_kappa_score(evaluation_df[y1], evaluation_df[y2])
    print(f"Cohen's kappa score")
    print(f"Person 1 and machine: {y1_k}, Person 2 and machine: {y2_k}, Between humans score{ between_humans_k}")

    return y1_k, y2_k, between_humans_k
    



def embed_reasoning(reasonings, tokenizer, embedding_model):
    """
    Takes an list of list where each list is multiple sentnces and returns and returns a pooled vector summarizing the sentence's embeddings.
    """
    
    if reasonings is None: 
        return None
    input = tokenizer(reasonings,return_tensors="pt", padding=True, truncation=True)
    
    with torch.no_grad():
        output = embedding_model(**input)
        return output.pooler_output ## 768


def compare_reasoning(tensors: List[torch.Tensor]) -> Tuple[int, torch.Tensor]: 
    """Computes the closest tensor to the mean of a list of tensors 

    Args:
        tensors (List[torch.Tensor]): Takes a list of tensors 

    Returns:
        Tuple[int, torch.Tensor]: Tensor closest to the mean 
    """
    mean_embedding = torch.mean(torch.stack(tensors), dim = 0)
    low_cos_sim = -1
    tensor_min_from_center = None
    index = 0
    
    for i, tensor in enumerate(tensors): 
        if tensor is None:
            continue
        cos_sim = torch.nn.functional.cosine_similarity(tensor.unsqueeze(0), mean_embedding.unsqueeze(0), dim=-1)
        if cos_sim.item() > low_cos_sim:
             low_cos_sim = cos_sim
             tensor_min_from_center = tensor
             index = i
             
    return index, tensor_min_from_center

sentence = "What is a dog?"


misc_tokens = ['```', ' ', '\n', '_', 'A', ':']

def compute_perplexity(input: any): 
    """Input is the output of the Langchain_core.messages.ai.AIMessage"""
    
    token_log_pair = [(item['token'], item['logprob'])for item in input.response_metadata['logprobs']['content']]
    log_vals = [val[1] for val in token_log_pair if val[0] not in misc_tokens]
    average = 1/len(log_vals)
    perplexity = np.exp(-average * (np.sum(np.asarray(log_vals))))
    return perplexity 


def self_consistency_gen(prompt, model, num_iterations): 
    pathways = []
    ai_output = []
    
    for i in range(num_iterations):
        output = model.invoke(prompt)
        if output.content is None: 
           pathways.append("No content") 
        else: 
            pathways.append(output.content)
        ai_output.append(output)

    return pathways, ai_output
        

def self_consistency_assess(pathways, langchain_output, embedding_model, tokenizer): 
    reasoning_paths = []
    count_paths = []
    perplexity = []
    
    for paths in pathways: 
        reasons = parse_val_from_json_string(paths, "reasons")
        counts = parse_val_from_json_string(paths, "count")

        # Naive method 
        reasoning_paths.append(reasons)
        count_paths.append(counts)
    
        
    for ai_output in langchain_output: 
        
        # Perplexity method 
        perplexity.append(compute_perplexity(ai_output))
    

    print(reasons)
    print(counts)
    print(perplexity)
        
    count_dicts = Counter(count_paths)
    most_common_number = count_dicts.most_common(1)[0][0]
    
    cache_reason = list(zip(count_paths, reasoning_paths, pathways))
    
    ### Find the worst reasoning paths 
    margin_paths  = [cache[1] for cache in cache_reason if cache[0] != most_common_number]
    
    ## Find the best reasoning path 
    dominant_paths = [cache[1] for cache in cache_reason if cache[0] == most_common_number]
    
    embedded_reasons_list = [embed_reasoning(reasonings=x, tokenizer=tokenizer, embedding_model=embedding_model) for x in dominant_paths]
    avg_embedded_reasons_list = [torch.mean(embed_group, dim=0) for embed_group in embedded_reasons_list]
    id, _ = compare_reasoning(avg_embedded_reasons_list)
    dom_reason = dominant_paths[id]
    
    
    perplexity_id = perplexity.index(min(perplexity))
    dom_r_perplex = dominant_paths[perplexity_id]

    
    return dom_reason, dom_r_perplex, perplexity, margin_paths


def safe_self_consistency(x, client, n):
    try:
        return pd.Series(self_consistency_gen(prompt = x, model=client, num_iterations=n))
    except Exception as e:
        print("Error processing row:", e)
        return pd.Series([None] * 2)
    
def self_consistency_assess_all(pathways, langchain_output):
    self_consist_dict = {
        "pathways": pathways,
        "langchain_output": langchain_output,
        "embedding_model": bert_model,
        "tokenizer": bert_tokenizer
        
    }
    try: 
        return pd.Series(self_consistency_assess(**self_consist_dict))
    except Exception as e:  
        print("Error assessing reasonings")
        return pd.Series([None] * 4)


def run_llm_model(model, 
                  df, 
                  df_col,
                  instructions): 
    sample_df = df.loc[(df['Q2_e'] == "Yes") & (df['Q2_s'] == "Yes") & (df['Q3_e'] <= 4) & (df['Q3_s'] <= 4)]
    sample_df.reset_index(drop=True, inplace=True)
    num = min(len(sample_df), 40)
    sample_df = sample_df.iloc[0:num]

    
    sample_df['prompt'] = sample_df.loc[:, df_col].apply(lambda x: construct_prompt(coding_instructions=instructions,
                                                                                                    few_shot=False,
                                                                                                    continuation_text=x))
    
    sample_df['json_output'] = sample_df['prompt'].apply(lambda x: model.invoke(x))
    sample_df['machine_count'] = sample_df['json_output'].apply(lambda x: parse_val_from_json_string(x.content, "count"))

    sample_df = sample_df.dropna(subset=['machine_count'])
    evaluate_matrix = sample_df[['machine_count', 'Q3_e', 'Q3_s']].astype(int)

    y1_k, y2_k, bet_k = compute_cohen_kappa(evaluate_matrix, 'machine_count', 'Q3_e', 'Q3_s')
        
    return sample_df, y1_k, y2_k, bet_k






### DEPRECATED: 
def self_consistency(prompt, 
                     model, 
                     embedding_model,
                     tokenizer,
                     num_iterations):
    """Self-consistency by Wei 2024 is the idea that we can simulate multiple reasoning paths and get 
    the most consistent values in order to have stable results. Self-consistency is however more 
    computationally expensive. 
    
    Step 1: Given a comprehensive prompt, simulate multiple rounds of answers
    Step 2: Append all outputs to pathways = [ ]
    Step 3: Extract the total count, the reasoning for those counts
    Step 4: cache_reason = list(zip together count, reasoning, paths) 
    Step 5: Extract the marginalized reasoning and paths for error inspection 
    Step 6: Extract the most common number 
    Step 7: Identify the best answers for the subset of most common number by taking cosine-similarity of the reasonings"""

      ### Run the model for a certain number of iterations
    # ret_df = df.copy()
    
    pathways = [] # Store entire model output 
    perplexity = [ ]
    for i in range(num_iterations): 
        output  = model.invoke(prompt)
        pathways.append(output.content)
        perplexity.append(compute_perplexity(output))
    
    ### Parse out reasoning paths and count
    reasoning_paths = [] # Store reasoning for inclusion criteria
    count_paths = [] # Store the number of count for the particular pathway 

    ### Identify the number of counts that appear most often and look at reasoning 
    for paths in pathways: 
        reasons = parse_val_from_json_string(paths, "reasons")
        counts = parse_val_from_json_string(paths, "count")
        
        reasoning_paths.append(reasons)
        count_paths.append(counts)
    
    count_dicts = Counter(count_paths)
    most_common_number = count_dicts.most_common(1)[0][0]
    
    cache_reason = list(zip(count_paths, reasoning_paths, pathways))
    ### Find the worst reasoning paths 
    margin_paths  = [cache[1] for cache in cache_reason if cache[0] != most_common_number]
    
    ## Find the best reasoning path 
    dominant_paths = [cache[1] for cache in cache_reason if cache[0] == most_common_number]
    embedded_reasons_list = [embed_reasoning(reasonings=x, tokenizer=tokenizer, embedding_model=embedding_model) for x in dominant_paths]
    avg_embedded_reasons_list = [torch.mean(embed_group, dim=0) for embed_group in embedded_reasons_list]
    id, _ = compare_reasoning(avg_embedded_reasons_list)
    dom_reason = dominant_paths[id]
    
    perplexity_id = perplexity.index(min(perplexity))
    dom_reason_perplex = dominant_paths[perplexity_id]
    

    ## Use dominant reason to index the JSON schema that best aligns with the output 
    dom_json_schema = [item[2] for item in cache_reason if item[1] == dom_reason]
    print(dom_json_schema)
    json_output = parse_json_string(dom_json_schema[0]) 
    print(json_output)

    
    return dom_reason, json_output, dom_reason_perplex, perplexity, most_common_number, margin_paths, cache_reason
    ## Keep track of all the reasoning paths, most common number of concepts extracted, and the marginal reasoning paths 
    # ret_df['machine_count_most'] = most_common_number
    # ret_df['cache_reason'] = cache_reason
    # ret_df['margin_paths'] = margin_paths






