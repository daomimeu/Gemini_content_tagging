import pandas as pd
import numpy as np
import re 

import google.generativeai as genai
import google.ai.generativelanguage as glm


from google.auth.transport.requests import Request
from google.api_core import retry
from google.oauth2 import service_account
from google.cloud import bigquery

from IPython.display import display
from IPython.display import Markdown
import textwrap
from tenacity import retry, stop_after_attempt, wait_fixed

import time
from tqdm import tqdm

import json

from google.generativeai.types import HarmCategory, HarmBlockThreshold



def fixJSON(jsonStr):
    # First remove the " from where it is supposed to be.
    jsonStr = re.sub(r'\\', '', jsonStr)
    jsonStr = re.sub(r'{"', '{`', jsonStr)
    jsonStr = re.sub(r'"}', '`}', jsonStr)
    jsonStr = re.sub(r'":"', '`:`', jsonStr)
    jsonStr = re.sub(r'":', '`:', jsonStr)
    jsonStr = re.sub(r'","', '`,`', jsonStr)
    jsonStr = re.sub(r'",', '`,', jsonStr)
    jsonStr = re.sub(r',"', ',`', jsonStr)
    jsonStr = re.sub(r'\["', '\[`', jsonStr)
    jsonStr = re.sub(r'"\]', '`\]', jsonStr)

    # Remove all the unwanted " and replace with ' '
    jsonStr = re.sub(r'"',' ', jsonStr)

    # Put back all the " where it supposed to be.
    jsonStr = re.sub(r'\`','\"', jsonStr)

    return json.loads(jsonStr)


def validate_gemini_answer(data_dict):
    def is_valid_answer_score(text):
        match = re.search(r'Score:\s*(\d+)', text)
        return bool(match)

    for k, v in data_dict.items():
        if isinstance(v, dict):
            if 'score' in v:
                if v['score'] is None or not isinstance(v['score'], int):
                    return False
            if 'answer' in v:
                if v['answer'] is None or not is_valid_answer_score(v['answer']):
                    return False
    return True


def generate_each_sj(subject_line, model, promt_json, success_lst, error_lst, 
                     question_curiosity, question_urgency, question_relavance, 
                     question_value, question_emotion, question_specificity, question_product_feature):
    retry_strategy = retry()  

    for attempt in range(10):
        try:
            response = model.generate_content(f"""
                Please analyse the subject line: {subject_line}. Please analyse for each criteria below.
                {question_curiosity}
                {question_urgency}
                {question_value}
                {question_relavance}
                {question_emotion}
                {question_specificity}
                {question_product_feature}
                Please note that the score given should be in 1 character. 
                """, request_options={'retry': retry_strategy})
            
            response_raw = response.text

            try:
                response = model.generate_content(textwrap.dedent(promt_json) + response_raw, generation_config={'response_mime_type':'application/json'})
            except Exception as e:
                time.sleep(120)
                response = model.generate_content(textwrap.dedent(promt_json) + response_raw, generation_config={'response_mime_type':'application/json'})
            
            json_data = response.text 

            try: 
                data_dict = json.loads(json_data)
                data_dict['subject_line'] = subject_line
            except json.JSONDecodeError as e:
                data_dict = fixJSON(e)
                data_dict['subject_line'] = subject_line
            
            if validate_gemini_answer(data_dict):  # Validate the answer and score
                success_lst.append(data_dict)
                return success_lst
            else:            
                print(f'{subject_line}: Missing score, retrying...')
                time.sleep(10)  

        except Exception as e:
            print(f'Retried {attempt + 1} times for{subject_line}')
            if attempt == 9:  
                error_lst.append({'subject_line': subject_line, 'error': 'Exceeded retries'})

    return success_lst, error_lst