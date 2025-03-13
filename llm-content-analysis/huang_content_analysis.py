from pydantic import BaseModel
from openai import OpenAI, OpenAIError
import random
import sys
from typing import List, Optional
import os
from dotenv import find_dotenv, load_dotenv
import pandas as pd 

dotenv_path = find_dotenv() ## Find the .env files
load_dotenv(dotenv_path=dotenv_path)

api_key = os.getenv("OPENAI_API_KEY")
OpenAI.api_key = api_key
client = OpenAI()

prompt = """
You are a world-renowned chef who specializes in nutrition. Write a breakfast plan for someone who is gluten free and paleo. It should be high in protein, healthy fats, and have good amount of carbs.
"""
## Models: 
## o-3 mini 

response = client.chat.completions.create(
    model="gpt-4o-mini-2024-07-18",
    messages=[
        {
            "role": "user", 
            "content": prompt
        }
    ],
    temperature=0.5,
    max_tokens = 2000
)

print(response.choices[0].message.content)

df = pd.DataFrame(columns=["prompt", "response"])

dct_res =  {"prompt": prompt, 
 "response": response.choices[0].message.content}

df = df._append(dct_res, ignore_index=True)


meta_content_ex = pd.read_excel('meta_final_data/meta_content_misinfo.xlsx')

c_a_prompt = """ "You are a research assistant at a top University. " \
           "Your task is to decide whether the given news releases satisfy all of the following criteria I1, I2, and I3\n"\
           "I1: The news must detail a proposed change associated with digital well-being, encompassing areas including but not limited to user safety, privacy, harmful content, misinformation, user behaviors pertinent to well-being, and/or age-appropriate design.\n"\
           "I2: The news must pertain to a change in platform products, features, policies, and/or the provision of information.\n" \
           "I3: The change must be either already implemented or scheduled for implementation by a social media platform.\n" \
           "Each news contains a title and a main content. " \
            "For example, see the following News:\n" \
            "Example news A -- Title: How We Approach Safety and Privacy in Community Chats; Content: Community Chats in Messenger and Facebook enhance connections by allowing real-time interactions through text, audio, or video around shared interests. These chats are part of Messenger's expansion to facilitate community-based public discussions, alongside private conversations. To ensure safety, admins can use robust moderation tools to manage content, automatically handle problematic messages, and maintain order. Users can report inappropriate interactions and block others. Privacy settings are respected, with options to manage visibility and message delivery. Meta is committed to improving data handling, content moderation, and the overall chat experience while adhering to Community Standards and providing user controls." \
           '\nNews A satisfies I1 and I2 and I3. ' \
            "\nExample news B  -- Title: YouTube gift codes are now available on Amazon; Content: This holiday season, YouTube gift codes are now available for purchase on Amazon in the U.S., offering a versatile gift option for entertainment. These codes can be used for YouTube Premium subscriptions, which provide ad-free viewing, offline playback, and background listening, or for YouTube Music Premium. They can also be applied to YouTube TV for access to over 70 live channels, movie rentals, and Channel Memberships, enhancing viewer support for creators. The codes are easy to buy and redeem, with email delivery options that include redemption instructions directly linked to the recipient's Google Play balance." \
            '\nNews B does not satisfy I1 and I2 and I3. ' \
            "\nExample news C -- Title: A collaborative approach to teen supervision on YouTube; Content: YouTube is launching a supervised experience that allows parents and teens to link their accounts for shared insights and notifications about channel activities, including uploads and comments. This new feature, part of the Family Center hub, aims to support responsible content creation and offers parents visibility into their teens’ YouTube interactions. It builds on existing services for younger users and includes guidance from experts in child development and digital learning. The initiative emphasizes mutual control between parents and teens, promoting informed and responsible digital citizenship while respecting teens' autonomy." \
            '\nNews C satisfies I1 and I2 and I3. ' \
            "\nExample news D -- Title:TikTok launches contest to take a Brazilian fan to Adele's concert in London.; Content: Competition within the platform will select the best video about the singer; the winner will have a VIP experience at the artist's concert with all expenses paid. Brazilian fans of Adele will not..." \
            '\nNews D does not satisfies I1 and I2 and I3. ' \
    '\nNow, please see the news release below. Does the news satisfy I1 and I2 and I3?' \
"\nPlease answer with the best of your capacity." \
            "Please only answer Yes or No and return your answer in json format.\nThe news release : " """


print(c_a_prompt)


### Pydantic base model is used for data validation to ensure lLM responses adhere to a priori determined structure

class LLMResponse(BaseModel):
    user_id: int
    prompt: str 
    response: List[str]
    
## So we have a dataframe of coded_responses which would get a tag of 1 (under coded_so_far)


"""

"""

few_shot_codes = [ ]
few_shot_texts = [ ]

def construct_prompt(coding_instructions: str, 
                     few_shot_texts: List[str], 
                     few_shot_codes: List[int], 
                     deductive_codes: int, 
                     continuation_text: str ):
    """Construct prompts for the dataframe, randomizes examples, randomizes the deductive codes to prevent LLM recency bias and 
    focus on internalizing the concept extraction tasks

    Args:
        coding_instructions (str): Manual instructions to type out 
        few_shot_texts (List[str]): A few text examples 
        few_shot_codes (List[int]): A few deductively coded examples for the corresponding texts
        deductive_codes (int): A list of codes to choose from
        dataframe (_type_): Dataframe of information 
    """
    ## Initial instruction 
    prompt = coding_instructions + "\n\n"
    
    ### Shuffle deductive codes to prevent recency bias in LLMs 
    deductive_codes = deductive_codes.copy()
    random.shuffle(deductive_codes)
    
    if not deductive_codes: 
        prompt += "Here are the following choices: \n\n"
        
        for code in deductive_codes:
            prompt += code + "\n"
    else:
        raise ValueError("No deductive codes to choose from. Please specify.")
    
    ### Shuffle examples of coded articles to prevent a few edge cases from dominating sample
    
    examples = list(zip(few_shot_texts, few_shot_codes))
    random.shuffle(examples)
    few_shot_texts, few_shot_codes = zip(*examples)
    
    ### Find the first four examples 
    for i in range(0, 4):
        prompts += "\n###\n\n Examples:" + few_shot_texts[i] + "\nAnswer:\n" + few_shot_codes[i]
    
    prompts += "\n\n" + continuation_text + "\nAnswer:\n"
    
    return prompts 

model="gpt-4o-mini-2024-07-18"

def get_llm_response(prompt: str, 
                     temperature: Optional[float] = 0.5,
                     max_tokens: Optional[int] = 1000) -> str: 
    """Takes the generated prompts and other LLM-model-related parameters to query OpenAI's API for an 
    LLM response
    
    Args:
        prompt (str): Question to query OpenAI's API
        temperature (float): Measure of 'randomness' of LLM's response
        max_tokens (int): Maximum number of tokens outputted in LLM's response
    """
    try: 
        response = OpenAI.chat.completions.create(
            model="gpt-4o-mini-2024-07-18",
            messages=[
                {
                    "role": "user", 
                    "content": prompt
                }
            ],
            temperature=temperature,
            max_tokens = max_tokens
        )
        
        llm_response = response.choices[0].message.content

        return llm_response  
    except OpenAIError as e:
        return f'Error: {e}'


    


    
     
    
    
    
    
    
            
        
        
        
    

    
    
    

    
    
    
