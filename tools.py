import pandas as pd 
from pydantic import BaseModel, Field
from typing import List, Optional, Annotated, Union, Tuple
from langchain_openai import ChatOpenAI
from langchain_core.messages.ai import AIMessage
from langchain_core.load.dump import dumpd
from langchain_core.load import * 
from langchain_core.prompts import ChatPromptTemplate
from dotenv import find_dotenv, load_dotenv
import random 
import json 
from sklearn.metrics import cohen_kappa_score
from collections import Counter
from sklearn.metrics.pairwise import cosine_similarity
from transformers import BertModel, BertTokenizer
import torch
import numpy as np



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


def str_to_json(response: str):
    try: 
        res = response.strip('```').strip('json')
        return json.loads(res)
        
    except json.JSONDecodeError as e: 
        print(f"Error decoding string into JSON output{e}")
        return None
    

def parse_json_string(response: str):
    """Decodes LLM output (str) into JSON object"""
    try: 
        res = response.strip('```').strip('json')
        print(res)
        return json.loads(res)
    
    except json.JSONDecodeError:
        print('Failed to decode JSON object.')
        return None 
    
def parse_val_from_json_string(json_string, key):
    """Extract output from JSON object such as "count", "actions" """

    try:  
        json_out = parse_json_string(json_string)
        if key in json_out:
            return json_out[key]
    
        if "actions" in json_out and isinstance(json_out['actions'], list):
            return [item[key] for item in json_out["actions"] if key in item]
        
        return f"Key '{key}' not found"
    except json.JSONDecodeError:
        return "Invalid JSON format"
    except Exception as e:
        return f"Error: {e}"


def compute_cohen_kappa(evaluation_df, machine_coder, y1, y2):
    """Passes in a dataframe of three columns with values of machine coders and two humans.
    Args:
        evaluation_df (pandas.DataFrame): pandas.DataFrame name 
        machine_coder (str): name of machine coder column
        y1 (str): name of human coder 1 column
        y2 (str): name of human coder 2 column 
    """
    evaluation_df = evaluation_df.copy()
    evaluation_df[machine_coder] = evaluation_df[machine_coder].astype(int)
    evaluation_df[y1] = evaluation_df[y1].astype(int)
    evaluation_df[y2] = evaluation_df[y2].astype(int)

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


misc_tokens = ['```', ' ', '\n', '_', 'A', ':']

def compute_perplexity(input: List[List[Tuple[str, int]]]): 
    """Input is a list of list of token, logprob pairs retrieved from Langchain AIMessage object."""
    
    ## Compute perplexity scores for all list of token-logprob pairs 
    
    perplexity_scores = []
    for i in range(len(input)): 
        log_vals = []
        for token, logprobs in input[i]: 
            if token not in misc_tokens: 
                log_vals.append(logprobs)
        
        n = len(log_vals)
        perplexity = np.exp(-1/n * (np.sum(np.asarray(log_vals))))
        perplexity_scores.append(perplexity )
        
    ## Calculate the minimum perplexity score 
    
    id = perplexity_scores.index(min(perplexity_scores))
    
    return id, perplexity_scores


def self_consistency_gen(prompt: str, model: any, num_iterations: int) -> Tuple[List, List]: 
    """Queries the LLM for a number of iterations 

    Args:
        prompt (str): Passes in the constructed prompt to query the model 
        model (any): Model object (e.g. chat-gpt-4o)
        num_iterations (int): Number of times to query the model 

    Returns:
        Tuple[List, List]: Returns a tuple of the model output and AIMessage object
    """
    output = []
    for _ in range(num_iterations):
        res = model.invoke(prompt)
        output.append(res if res else "No content")
        
    return output 


def parse_langchain_output(ai_message: List[AIMessage]) -> Tuple[Tuple, str, int]: 
    """Takes a LangChain AIMessage object and returns a tuple of token_log_pairs, 
    model response, and total tokens used 

    Args:
        ai_message (AIMessage): Langchain AIMessage object
    Raises:
        TypeError: If input is not a Langchain AIMessage object

    Returns:
        Tuple: Response and other metadata information
    """
    
    if not isinstance(ai_message, AIMessage):
            raise TypeError(f'Input is not Langchain AIMessage object, it is {type(ai_message)}')
        
    ### Retrieve logprobs 
    token_log_pair = [(item['token'], item['logprob'])for item in ai_message.response_metadata['logprobs']['content']]
    
    ### Retrieve content of message 
    content = ai_message.content
    
    ### Retrieve total token use (prompt and response)
    token_usage = ai_message.response_metadata['token_usage']['total_tokens']

    return token_log_pair, content, token_usage


def assess_self_consistency_perplexity(message_list): 

  zipped_outputs = [parse_langchain_output(message) for message in message_list]
  token_log_pairs, res, token_use = zip(*zipped_outputs)
  
  total_token_use = sum(token_use)
  print(total_token_use)
  
  count_list = [parse_val_from_json_string(rs, "count") for rs in res]
  print(type(res[0]))
  print(count_list)
  
  cache_reason = list(zip(token_log_pairs, count_list, res))
  
  print(len(cache_reason))
  
  count_dict = Counter(count_list)
  
  most_common_n = count_dict.most_common(1)[0][0]
  
  
  relevant_t_l_pairs = [log_pairs for log_pairs, count, _ in cache_reason if count == most_common_n]
  margin_paths = [res for _, count, res in cache_reason if count != most_common_n]
  
  id_min, perplexity_list = compute_perplexity(relevant_t_l_pairs)
  
  min_perplexity_score = perplexity_list[id_min]
  
  _, _, best_response = cache_reason[id_min]
  
  return best_response, perplexity_list, most_common_n, margin_paths, total_token_use

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
    
    count_dicts = Counter(count_paths)
    most_common_number = count_dicts.most_common(1)[0][0]
    
    cache_reason = list(zip(count_paths, reasoning_paths, pathways, perplexity))
    
    ### Find the worst reasoning paths 
    margin_paths  = [cache[1] for cache in cache_reason if cache[0] != most_common_number]
    
    ## Find the best reasoning path 
    
    dominant_paths = [cache[1] for cache in cache_reason if cache[0] == most_common_number]
    
    embedded_reasons_list = [embed_reasoning(reasonings=x, tokenizer=tokenizer, embedding_model=embedding_model) for x in dominant_paths]
    avg_embedded_reasons_list = [torch.mean(embed_group, dim=0) for embed_group in embedded_reasons_list]
    id, _ = compare_reasoning(avg_embedded_reasons_list)
    dom_reason = dominant_paths[id]
    
    _, dom_perplex = min(enumerate(dominant_paths), key=lambda x: x[1][3])
    
    dom_r_perplex = dom_perplex[1][0]
    # perplexity_id = [cache[4] for cache in cache_reason if cache[0] == most_common_number]
    
    # dom_r_perplex = dominant_paths[perplexity_id]
    
    if not dom_reason or not dom_r_perplex or not most_common_number: 
        print("Inspect reasoning")
        dom_reason = "N/A"
        dom_r_perplex = "N/A"
        most_common_number = 999
        
    if not perplexity: 
        print('Perplexity list is empty')
    
    if not margin_paths:
        margin_paths = "N/A"
        
    
    return dom_reason, dom_r_perplex, perplexity, margin_paths, most_common_number


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

        ## Input validation
        if not isinstance(pathways, list):
            raise ValueError(f"Pathways must be list-like, got {type(pathways)}")
        if not pathways:
            raise ValueError(f"Pathways is empty")

        if not isinstance(langchain_output, list):
            raise TypeError(f"Langchain output is not in a list")
        
        for i, item in enumerate(langchain_output):
            if not isinstance(item, AIMessage):
                raise TypeError(
                    f"Individual langchain outputs not individual AIMessage instances."
                    f"Found {type(item).__name__} at index {i}. "
                    )

        return pd.Series(self_consistency_assess(**self_consist_dict))
    except Exception as e:  
        print(f"Error assessing reasonings{e}")
        return pd.Series([None] * 5)


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



def custom_serialization(ai_messages):
  serialized = []
  
  for message in ai_messages: 
    serialized.append(dumpd(message))
    
  return serialized
  

def recover_langchain_obj(serialized_objects):
  langchain_class = []
  
  for obj in serialized_objects: 
    langchain_class.append(load(obj))
    
  return langchain_class

def manage_files_safely(export, filename: str, langchain_output_col: Optional[str], data: Optional[any]=None,): 
  # Warning: AIMessage langchain objects are not recoverable when an entire dataframe is exported 
  # into Excel file 
  if export is True:
    df = data.copy()
  
    # Make Langchain class serializable 
    df[langchain_output_col] = df[langchain_output_col].apply(custom_serialization) 
  
    # Convert dataframe to a dictionary to save as JSON file 
    df_dict = df.to_dict(orient='records')

    # Export as JSON file   
    with open(filename, "w") as file: 
        json.dump(df_dict, file)
        print("Done exporting as JSON file...")
    # Test to see if the JSON file is a success
    
  with open(filename, "r") as file: 
        ret_dict = json.load(file)
        print("Done re-importing JSON file...")
        
    # Recover the Langchain object 
  ret_df = pd.DataFrame(ret_dict)
  ret_df[langchain_output_col] = ret_df[langchain_output_col].apply(recover_langchain_obj)

  print(ret_df[langchain_output_col])

  return ret_df 
  



### DEPRECATED: 
# def self_consistency(prompt, 
#                      model, 
#                      embedding_model,
#                      tokenizer,
#                      num_iterations):
#     """Self-consistency by Wei 2024 is the idea that we can simulate multiple reasoning paths and get 
#     the most consistent values in order to have stable results. Self-consistency is however more 
#     computationally expensive. 
    
#     Step 1: Given a comprehensive prompt, simulate multiple rounds of answers
#     Step 2: Append all outputs to pathways = [ ]
#     Step 3: Extract the total count, the reasoning for those counts
#     Step 4: cache_reason = list(zip together count, reasoning, paths) 
#     Step 5: Extract the marginalized reasoning and paths for error inspection 
#     Step 6: Extract the most common number 
#     Step 7: Identify the best answers for the subset of most common number by taking cosine-similarity of the reasonings"""

#       ### Run the model for a certain number of iterations
#     # ret_df = df.copy()
    
#     pathways = [] # Store entire model output 
#     perplexity = [ ]
#     for i in range(num_iterations): 
#         output  = model.invoke(prompt)
#         pathways.append(output.content)
#         perplexity.append(compute_perplexity(output))
    
#     ### Parse out reasoning paths and count
#     reasoning_paths = [] # Store reasoning for inclusion criteria
#     count_paths = [] # Store the number of count for the particular pathway 

#     ### Identify the number of counts that appear most often and look at reasoning 
#     for paths in pathways: 
#         reasons = parse_val_from_json_string(paths, "reasons")
#         counts = parse_val_from_json_string(paths, "count")
        
#         reasoning_paths.append(reasons)
#         count_paths.append(counts)
    
#     count_dicts = Counter(count_paths)
#     most_common_number = count_dicts.most_common(1)[0][0]
    
#     cache_reason = list(zip(count_paths, reasoning_paths, pathways))
#     ### Find the worst reasoning paths 
#     margin_paths  = [cache[1] for cache in cache_reason if cache[0] != most_common_number]
    
#     ## Find the best reasoning path 
#     dominant_paths = [cache[1] for cache in cache_reason if cache[0] == most_common_number]
#     embedded_reasons_list = [embed_reasoning(reasonings=x, tokenizer=tokenizer, embedding_model=embedding_model) for x in dominant_paths]
#     avg_embedded_reasons_list = [torch.mean(embed_group, dim=0) for embed_group in embedded_reasons_list]
#     id, _ = compare_reasoning(avg_embedded_reasons_list)
#     dom_reason = dominant_paths[id]
    
#     perplexity_id = perplexity.index(min(perplexity))
#     dom_reason_perplex = dominant_paths[perplexity_id]
    

#     ## Use dominant reason to index the JSON schema that best aligns with the output 
#     dom_json_schema = [item[2] for item in cache_reason if item[1] == dom_reason]
#     print(dom_json_schema)
#     json_output = parse_json_string(dom_json_schema[0]) 
#     print(json_output)

    
#     return dom_reason, json_output, dom_reason_perplex, perplexity, most_common_number, margin_paths, cache_reason
#     ## Keep track of all the reasoning paths, most common number of concepts extracted, and the marginal reasoning paths 
#     # ret_df['machine_count_most'] = most_common_number
#     # ret_df['cache_reason'] = cache_reason
#     # ret_df['margin_paths'] = margin_paths






