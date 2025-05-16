import pandas as pd 
from pydantic import BaseModel, Field
from typing import List, Optional, Annotated, Union, Tuple, Callable, Any
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

def construct_prompt(instructions: str, 
                     few_shot: bool, 
                     continuation_texts: str, 
                     few_shot_texts: Optional[List[str]] = None, 
                     few_shot_answers: Optional[List[int]] = None, 
                     few_shot_num: Optional[int] = 3) -> str:
    """Takes prompt instructions, a question, and optional few-shot examples, randomizes the few-shot examples,
    adds contextual calibration into a fully constructed prompt. 

    Args:
        instructions (str): Prompt 
        few_shot (bool): If prompt provides few shot examples 
        continuation_texts (str): Question
        few_shot_texts (Optional[List[str]], optional): A list of few shot examples. Defaults to None.
        few_shot_answers (Optional[List[int]], optional): A list of few shot answers. Defaults to None.
        few_shot_num (Optional[int], optional): How many few shot examples to use? Defaults to 3.

    Returns:
        full_prompt (str): A fully constructed prompt that can be used as input for language model.
    """
    
    
    prompt = instructions.strip() + "\n\n"
    
    prompt_structure = ChatPromptTemplate.from_template(prompt + "{exemplar_icl}" + "{text}" + "\n\nAnswer:\n")

    if few_shot == True: 
        
        ## Shuffle codes to prevent recency bias (Zhao et. al 2021)
        examples = list(zip(few_shot_texts, few_shot_answers))
        random.shuffle(examples)
        few_shot_texts, few_shot_answers = zip(*examples)
        
        few_shot_examples =""
        
        for i in range(min(few_shot_num, len(few_shot_texts))): 
            few_shot_examples += "\n\n###Examples:\n" + few_shot_texts[i] + "\n\n###Answer:\n" + few_shot_answers[i]

        ## Optional contextual calibration (Zhao et al. 2021)
        few_shot_examples += "\n\n###Examples:\n" + "N/A [MASK] [MASK]" + "\n\n###Answer:\n" + "N/A" + "\n\n"
            
            
    else: 
        few_shot_examples = " "
    
    full_prompt = prompt_structure.format(exemplar_icl=few_shot_examples, text=continuation_texts)
        
    return full_prompt



def generate_json_str_task_1(codes: Tuple[Any, Any, Any]= None) -> Union[str, Any]:
    """Generate example outputs of JSON string given a list of few shot example human-labels for task 1 (estimating number of design)

    Args:
        codes (Tuple[Any, Any, Any], optional): A tuple of COUNTS (sum of digital design codes), SHORT DESCRIPTIONS of each design, and REASON for why design satisfies inclusion criteria. Defaults to None.

    Raises:
        KeyError: if the length of descriptions or reasons does not match the COUNT
        KeyError: if the length of descriptions or reasons does not match the COUNT

    Returns:
        Union[str, Any]: _description_
    """
    if not codes:
        return "N/A"
    
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


def str_to_json(response: str) -> Any:
    """Parse JSON string and convert back to JSON object. Takes an input and decode string into JSON output.

    Args:
        response (str): A JSON-formatted string

    Returns:
        _type_: A JSON object or none if the parsing fails. 
    """
    try: 
        res = response.strip('```').strip('json')
        return json.loads(res)
        
    except json.JSONDecodeError as e: 
        print(f"Error decoding string into JSON output{e}")
        return None
    

## We can comment this out
def parse_json_string(response: str):
    """Decodes LLM output (str) into JSON object"""
    try: 
        res = response.strip('```').strip('json')
        print(res)
        return json.loads(res)
    
    except Exception as e:
        print(f'Failed to decode JSON object due to {e}')
        return None 
    
    
def parse_val_from_json_string(json_string: str, 
                               key: str) -> Any:
    """Parse a JSON string and extract the value(s) associated with a given key.

    If the key is found at the top level of the JSON object, its value is returned.
    If not, but the key is found in any item within a list under the "actions" key,
    a list of matching values is returned.
    If the key is not found at all, a message is returned.

    Args:
        json_string (str): The input JSON string to parse.
        key (str): The key to look for in the JSON object.

    Returns:
        Any: The value(s) associated with the key, a not-found message, or an error message.
    """
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


def compute_cohen_kappa(evaluation_df: pd.DataFrame,
                        machine_coder: str, 
                        y1: str, 
                        y2: str) -> Tuple[float, float, float]:
    """Compute Cohen's kappa scores between a machine coder and two human coders,
    as well as the agreement between the two human coders.
    
    Args:
        evaluation_df (pd.DataFrame): A DataFrame containing the annotations
        machine_coder (str): Column name for machine-generated labels
        y1 (str): Column name of first human coder's label
        y2 (str): Column name of second human coder's label

    Returns:
        Tuple[float, float, float]: 
            - Cohen's kappa between machine and human 1,
            - Cohen's kappa between machine and human 2,
            - Cohen's kappa between human 1 and human 2.
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
    

model_name = 'bert-base-uncased'
bert_tokenizer = BertTokenizer.from_pretrained(model_name)
bert_model = BertModel.from_pretrained(model_name)
bert_model.eval()


def embed_reasoning(reasonings: str, 
                    tokenizer: Any, 
                    embedding_model: Any) -> Union[None, torch.Tensor]:
    """Embeds reasoning text using a transformer model and returns a pooled vector.


    Args:
        reasonings (str): Input text (single string or multiple sentences concatenated).
        tokenizer (Any): HuggingFace tokenizer for the embedding model.
        embedding_model (Any):  Pretrained transformer model that outputs embeddings.

    Returns:
        Union[None, torch.Tensor]: A pooled embedding tensor (usually size [1, 768]) or None if input is None.
    """
    if reasonings is None: 
        return None
    input = tokenizer(reasonings,return_tensors="pt", padding=True, truncation=True)
    
    with torch.no_grad():
        output = embedding_model(**input)
        return output.pooler_output ## 768


def compare_reasoning(tensors: List[torch.Tensor]) -> Tuple[int, torch.Tensor]: 
    """Given a list of tensor, this function identifies the closest pooled vector to the average of the vectors using cosine similarity

    Args:
        tensors (List[torch.Tensor]): A list of tensors size typically [1, 768]

    Returns:
        Tuple[int, torch.Tensor]: 
            - Index of the tensor closest to the mean embedding,
            - The corresponding tensor itself.
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

def compute_perplexity(input: List[List[Tuple[str, int]]]) -> Tuple[int,List[float]]: 
    """
    Computes perplexity scores from token-logprob pairs, and returns the index
    of the sequence with the lowest perplexity (i.e., most fluent).

    Args:
        input (List[List[Tuple[str, float]]]): A list where each element is a list of (token, logprob) tuples.

    Returns:
        Tuple[int, List[float]]: 
            - Index of the sequence with lowest perplexity.
            - List of perplexity scores for all sequences.
    """
    perplexity_scores = []
    for i in range(len(input)): 
        log_vals = []
        for token, logprobs in input[i]: 
            if token not in misc_tokens: 
                log_vals.append(logprobs)
        
        n = len(log_vals)
        perplexity = np.exp(-1/n * (np.sum(np.asarray(log_vals))))
        perplexity_scores.append(perplexity)
        
    ## Calculate the minimum perplexity score 
    
    id = perplexity_scores.index(min(perplexity_scores))
    
    return id, perplexity_scores


def self_consistency_gen(prompt: str, model: Any, num_iterations: int) -> Tuple[List, List]: 
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
        Tuple[Tuple, str, str]: Response and other metadata information
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


def assess_self_consistency_perplexity(message_list: List[AIMessage]) -> Tuple[str, List[float], int, List[str], int]: 
    """Evaluates self-consistency in reasoning by identifying the best reasoning path 
    based on perplexity scores and the most common 'count' value.

    Args:
        message_list (List[str]): A list of Langchain message outputs containing token-logprob pairs.

    Returns:
        Tuple[str, List[float], int, List[str], int]: 
            - Best response (string) with lowest perplexity.
            - List of perplexity scores for all sequences.
            - The most common count value.
            - List of paths with non-common count.
            - Total token use across all messages.
    """

    zipped_outputs = [parse_langchain_output(message) for message in message_list]
    token_log_pairs, res, token_use = zip(*zipped_outputs)
    
    total_token_use = sum(token_use)
    
    count_list = [parse_val_from_json_string(rs, "count") for rs in res]

    
    cache_reason = list(zip(token_log_pairs, count_list, res))
    
    count_dict = Counter(count_list)
    
    most_common_n = count_dict.most_common(1)[0][0]
    
    
    relevant_t_l_pairs = [log_pairs for log_pairs, count, _ in cache_reason if count == most_common_n]
    margin_paths = [res for _, count, res in cache_reason if count != most_common_n]
    
    id_min, perplexity_list = compute_perplexity(relevant_t_l_pairs)
    
    min_perplexity_score = perplexity_list[id_min]
    
    _, _, best_response = cache_reason[id_min]
    
    return best_response, perplexity_list, most_common_n, margin_paths, total_token_use
  

def custom_serialization(ai_messages: List[AIMessage]) -> str:
    """Takes the output of a Langchain query and converts it into a 

    Args:
        ai_messages (AIMessage): _description_

    Returns:
        str: _description_
    """
    serialized = []
  
    for message in ai_messages: 
        serialized.append(dumpd(message))
    
    return serialized
  

def recover_langchain_obj(serialized_objects: str) -> List[AIMessage]:
    """Recover AIMessage object from serialized objects (or JSON string)

    Args:
        serialized_objects (str): А list of formatted JSON string originally parsed from a Langchain object AIMessage

    Returns:
        List[AIMessage]: A list of recovered Langchain AIMessage object
    """
    langchain_class = []
    
    for obj in serialized_objects: 
        langchain_class.append(load(obj))
        
    return langchain_class

def manage_files_safely(export: bool, 
                        filename: str, 
                        langchain_output_col: Optional[str], 
                        data: Optional[Any] = None) -> Optional[pd.DataFrame]:
    """
    Safely exports a DataFrame containing Langchain objects to JSON and re-imports it.
    Ensures custom serialization and recovery of Langchain objects.

    Args:
        export (bool): Whether to export the data or just load from file.
        filename (str): The JSON filename to save to or load from.
        langchain_output_col (Optional[str]): The column containing Langchain AIMessage objects.
        data (Optional[Any]): The DataFrame to export, required if export=True.

    Returns:
        Optional[pd.DataFrame]: The DataFrame with recovered Langchain objects.
    """

    if export:
        if data is None or langchain_output_col is None:
            print("Export requested but data or langchain_output_col not provided.")
            return None
        
        df = data.copy()

        # Make Langchain class serializable
        df[langchain_output_col] = df[langchain_output_col].apply(custom_serialization)

        # Convert to dictionary and export as JSON
        df_dict = df.to_dict(orient='records')
        with open(filename, "w") as file:
            json.dump(df_dict, file)
            print(f"Exported JSON to {filename}")

    # Load from file
    with open(filename, "r") as file:
        ret_dict = json.load(file)
        print(f"Loaded JSON from {filename}")

    # Recover Langchain objects
    ret_df = pd.DataFrame(ret_dict)
    if langchain_output_col:
        ret_df[langchain_output_col] = ret_df[langchain_output_col].apply(recover_langchain_obj)
        print(f"Recovered Langchain objects in column '{langchain_output_col}'")

    return ret_df
  
  
def safe_self_consistency(x, client, n):
    try:
        return pd.Series(self_consistency_gen(prompt = x, model=client, num_iterations=n))
    except Exception as e:
        print("Error processing row:", e)
        return pd.Series([None] * 2)
    
def map_str_task_to_label(json_str, key, mapping): 
  parsed_dict = parse_json_string(json_str)
  mapped_list = []
  if parsed_dict: 
    if key in parsed_dict: 
      for item in parsed_dict[key]: 
        mapped_list.append([mapping[sub_item] for sub_item in list(item.values())[0]])
      return mapped_list  
    else: 
      return None
  else: 
    return None

def combine_downstream_json_task(json_str_base: str, downstream_task: str, key: str, mapping: dict) -> Any: 
  """Combine downstream LLM output with the initial base JSON output 

  Args:
      json_str_base (str): The serialized task of the base task. 
      downstream_task (str): The serialized (i.e. JSON-formatted string) JSON output from a downstream content analysis task 
      key (str): The key of the downstream task's dictionary.
      mapping (_type_): The dictionary to map the categorization to machine coded categorization into numeric labels
  
  Returns: 
      Returns a deserialized and modified dictionary. You can use json.dumps to overwrite the initial seralized JSON output. 
  """
  
  base_dict = parse_json_string(json_str_base)
  downstream_task_out = map_str_task_to_label(downstream_task, key, mapping)
  
  if base_dict and downstream_task_out: 
    for i, item in enumerate(list(base_dict.values())[0]):
          item[key] = downstream_task_out[i]
          
  else: 
    return None
    
  return base_dict

def jaccard_similarity(a, b):
    a_set, b_set = set(a), set(b)
    
    if not a_set and not b_set:
        
        return 1.0  # edge case: both empty sets
    
    return len(a_set & b_set) / len(a_set | b_set)


def soft_matching(
    output_list: List[Tuple[str, Union[int, set]]],
    comparison_list: List[Tuple[str, Union[int, set]]],
    threshold,
    tokenizer = bert_tokenizer,
    model = bert_model
) -> float:
    """
    Computes soft matches of texts using cosine similarity, and evaluates category
    similarity (Jaccard) for similar text pairs.

    Args:
        output_list: List of (text, category) tuples.
        comparison_list: List of (text, category) tuples.
        bert_tokenizer: Pretrained tokenizer.
        bert_model: Pretrained BERT model.

    Returns:
        float: Mean Jaccard similarity for matched text pairs.
    """
    if not output_list or not comparison_list:
        return 0.0
    else: 
        embedded_vector = [embed_reasoning(text, tokenizer, model) for text, _ in output_list]
        comparison_vector = [embed_reasoning(text, tokenizer, model) for text, _ in comparison_list]
        jaccard_scores = []
        for i, emb1 in enumerate(embedded_vector):
            for j, emb2 in enumerate(comparison_vector):
                cos_sim = torch.nn.functional.cosine_similarity(emb1, emb2).item()
                if cos_sim > threshold:
                    jac_sim = jaccard_similarity(output_list[i][1], comparison_list[j][1])
                    jaccard_scores.append(jac_sim)
        
        return np.mean(jaccard_scores) if jaccard_scores else 0.0

    
   
def within_observation_soft_matching(df, col_machine, col_hc_1, col_hc_2, threshold, tokenizer, embedding_model, exclude_null=True):
    """Soft-matching design changes within articles"""
    dataset = df.where(pd.notna(df), None).copy()
    print('Step 1---------')
    
    if exclude_null: 
        dataset=dataset.dropna(subset=[col_machine, col_hc_1])
    dataset['jaccard_score_hc1'] = dataset.apply(
        lambda row: soft_matching(output_list=row[col_machine],
                                    comparison_list=row[col_hc_1],
                                    threshold=threshold,
                                    tokenizer=tokenizer,
                                    model=embedding_model), axis=1)
    jc1 = dataset['jaccard_score_hc1'].mean()
    
    dataset = df.where(pd.notna(df), None).copy()
    print('Step 2---------')
    
    if exclude_null: 
        dataset=dataset.dropna(subset=[col_machine, col_hc_2])
    dataset['jaccard_score_hc2'] = dataset.dropna(subset=[col_machine, col_hc_2]).apply(
        lambda row: soft_matching(output_list=row[col_machine],
                                    comparison_list=row[col_hc_2],
                                    threshold=threshold,
                                    tokenizer=tokenizer,
                                    model=embedding_model), axis=1)
    
    jc2 = dataset['jaccard_score_hc2'].mean()

    dataset = df.where(pd.notna(df), None).copy()
    print('Step 3---------')
    
    if exclude_null: 
        dataset=dataset.dropna(subset=[col_hc_1, col_hc_2])
    dataset['jaccard_score_inter'] = dataset.dropna(subset=[col_hc_1, col_hc_2]).apply(
        lambda row: soft_matching(output_list=row[col_hc_1],
                                  comparison_list=row[col_hc_2],
                                  threshold=threshold,
                                  tokenizer=tokenizer,
                                  model=embedding_model), axis=1)
    
    jc3 = dataset['jaccard_score_inter'].mean()

    
    print(f'Jaccard similarity between machine and first HC: {jc1}')
    print(f'Jaccard similarity between machine and second HC: {jc2}')
    print(f'Jaccard similarity between human coders: {jc3}')
    
    return dataset, jc1, jc2, jc3
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




# def self_consistency_assess(pathways, langchain_output, embedding_model, tokenizer): 
#     reasoning_paths = []
#     count_paths = []
#     perplexity = []
    
#     for paths in pathways: 
#         reasons = parse_val_from_json_string(paths, "reasons")
#         counts = parse_val_from_json_string(paths, "count")
        
#         # Naive method 
#         reasoning_paths.append(reasons)
#         count_paths.append(counts)
    
#     for ai_output in langchain_output: 
#         # Perplexity method 
#         perplexity.append(compute_perplexity(ai_output))
    
#     count_dicts = Counter(count_paths)
#     most_common_number = count_dicts.most_common(1)[0][0]
    
#     cache_reason = list(zip(count_paths, reasoning_paths, pathways, perplexity))
    
#     ### Find the worst reasoning paths 
#     margin_paths  = [cache[1] for cache in cache_reason if cache[0] != most_common_number]
    
#     ## Find the best reasoning path 
    
#     dominant_paths = [cache[1] for cache in cache_reason if cache[0] == most_common_number]
    
#     embedded_reasons_list = [embed_reasoning(reasonings=x, tokenizer=tokenizer, embedding_model=embedding_model) for x in dominant_paths]
#     avg_embedded_reasons_list = [torch.mean(embed_group, dim=0) for embed_group in embedded_reasons_list]
#     id, _ = compare_reasoning(avg_embedded_reasons_list)
#     dom_reason = dominant_paths[id]
    
#     _, dom_perplex = min(enumerate(dominant_paths), key=lambda x: x[1][3])
    
#     dom_r_perplex = dom_perplex[1][0]
#     # perplexity_id = [cache[4] for cache in cache_reason if cache[0] == most_common_number]
    
#     # dom_r_perplex = dominant_paths[perplexity_id]
    
#     if not dom_reason or not dom_r_perplex or not most_common_number: 
#         print("Inspect reasoning")
#         dom_reason = "N/A"
#         dom_r_perplex = "N/A"
#         most_common_number = 999
        
#     if not perplexity: 
#         print('Perplexity list is empty')
    
#     if not margin_paths:
#         margin_paths = "N/A"
        
    
#     return dom_reason, dom_r_perplex, perplexity, margin_paths, most_common_number


# def self_consistency_assess_all(pathways, langchain_output):
#     self_consist_dict = {
#         "pathways": pathways,
#         "langchain_output": langchain_output,
#         "embedding_model": bert_model,
#         "tokenizer": bert_tokenizer
        
#     }
#     try: 

#         ## Input validation
#         if not isinstance(pathways, list):
#             raise ValueError(f"Pathways must be list-like, got {type(pathways)}")
#         if not pathways:
#             raise ValueError(f"Pathways is empty")

#         if not isinstance(langchain_output, list):
#             raise TypeError(f"Langchain output is not in a list")
        
#         for i, item in enumerate(langchain_output):
#             if not isinstance(item, AIMessage):
#                 raise TypeError(
#                     f"Individual langchain outputs not individual AIMessage instances."
#                     f"Found {type(item).__name__} at index {i}. "
#                     )

#         return pd.Series(self_consistency_assess(**self_consist_dict))
#     except Exception as e:  
#         print(f"Error assessing reasonings{e}")
#         return pd.Series([None] * 5)
