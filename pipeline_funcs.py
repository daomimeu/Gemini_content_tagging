import utils.gemini_utils as gm
import utils.db_dedup as db_dedup 

import pandas as pd
import numpy as np

import google.generativeai as genai
import google.ai.generativelanguage as glm
from google.generativeai.types import HarmCategory, HarmBlockThreshold


from google.auth.transport.requests import Request
from google.oauth2 import service_account
from google.cloud import bigquery

from IPython.display import display
from IPython.display import Markdown

from tqdm import tqdm
import json
import emoji



#Gemini prompt:
question_curiosity = 'From 0 to 5, rate the curiosity for this subject line. Please explain with Strengths, Weaknesses, improvements.'
question_urgency = 'From 0 to 5, rate the urgency for this subject line. Please explain with Strengths, Weaknesses, improvements.'
question_relavance = 'From 0 to 5, rate the subject line on mentioning trending topics, technologies or events. Please explain with Strengths, Weaknesses, improvements.'
question_value = 'From 0 to 5, rate the monetary value for this subject line. (Does this subject line mention value or prize?). The explanation should include strengths, weaknesses, improvements.'
question_emotion = 'From 0 to 5, rate the emotion for this subject line. Please explain with Strengths, Weaknesses, improvements.'
question_specificity = 'From 0 to 5, rate the specificity for this subject line. Please explain with Strengths, Weaknesses, improvements.'
question_product_feature = """Which technical product features do this subject line mention? Only answer a list separated by |. 
                            PLEASE NOTE THAT If there is no smartphone product features, please reponse as text null"""


#JSON format
promt_json = """
      Please return VALID JSON describing the score and answer for each criterias. Please follow the following shema and note that \n means line breaks.
      {   
    "Curiosity": {
        "score": "int",
        "answer": r"Score: \nStrengths: \nWeaknesses: \nImprovements:"},
    "Urgency": {
        "score": "int",
        "answer": r"Score: \nStrengths: \nWeaknesses: \nImprovements:"},
    "Trending Topics/Technologies": {
        "score": "int",
        "answer": r"Score: \nStrengths: \nWeaknesses: \nImprovements:"},
    "Monetary Value": {
        "score": "int",
        "answer": r"Score: \nStrengths: \nWeaknesses: \nImprovements:"},
    "Emotion": {
        "score": "int",
        "answer": r"Score: \nStrengths: \nWeaknesses: \nImprovements:"},
    "Specificity": {
        "score": "int",
        "answer": r"Score: \nStrengths: \nWeaknesses: \nImprovements:"},
    "product_features": "str"
}

  
      All fields are required. Please DON'T MISS ANY ANSWER, DON'T miss any value.

      Important: Only return a single piece of valid JSON text. Don't return invalid JSON with error PLEASE.

      Here is the raw response:

      """


### FUNCTIONS

def config_gemini(gemini_api_key_path, gemini_model):
    f = open(gemini_api_key_path, "r")
    gemini_api_key = json.loads(f.read())['key']
    genai.configure(api_key = gemini_api_key)
    model = genai.GenerativeModel(
        gemini_model,
        safety_settings= #Disable safety settings in Gemini. Ref: https://stackoverflow.com/questions/77947637/how-to-disable-safety-settings-in-gemini-vision-pro-model-using-api
    [
        {
            "category": "HARM_CATEGORY_HARASSMENT",
            "threshold": "BLOCK_NONE",
        },
        {
            "category": "HARM_CATEGORY_HATE_SPEECH",
            "threshold": "BLOCK_NONE",
        },
        {
            "category": "HARM_CATEGORY_SEXUALLY_EXPLICIT",
            "threshold": "BLOCK_NONE",
        },
        {
            "category": "HARM_CATEGORY_DANGEROUS_CONTENT",
            "threshold": "BLOCK_NONE",
        },
    ],)

    return model 


def get_sl_from_bq(service_account_json):

    credentials = service_account.Credentials.from_service_account_file(service_account_json)
    client = bigquery.Client(project = 'xxx', credentials=credentials)

    QUERY_NEW_SL = (
            '''
        SELECT DISTINCT subject_line
        FROM 
        (
            SELECT Email_Title AS subject_line, date
            FROM `xxx.gcdm.campaigns`
            WHERE Channel = 'EMAIL'
            
            UNION ALL

            SELECT title AS subject_line, date
            FROM `xxx.gcdm.campaign_asset_push` a
            JOIN `xxx.gcdm.campaigns` p
            ON a.HYBRIS_ID = p.HYBRIS_ID

            UNION ALL

            SELECT ticker AS subject_line, date
            FROM `xxx.gcdm.campaign_asset_push` a
            JOIN `xxx.gcdm.campaigns` p
            ON a.HYBRIS_ID = p.HYBRIS_ID
        )        
        '''
        )

    ### For Flagship campaigns only
    # FLAGSHIP_QUERY_NEW_SL = (
    #         '''
    #     SELECT DISTINCT subject_line
    #     FROM 
    #     (
    #         SELECT Email_Title AS subject_line, date
    #         FROM `xxx.DASHBOARD_FLAGSHIP.Q6B6_CAMPAIGN_PERFORMANCE_EDM`

    #         UNION ALL

    #         SELECT title AS subject_line, date
    #         FROM `xxx.gcdm.campaign_asset_push` a
    #         JOIN `xxx.DASHBOARD_FLAGSHIP.Q6B6_CAMPAIGN_PERFORMANCE_PUSH` p
    #         ON a.HYBRIS_ID = p.HYBRIS_ID

    #         UNION ALL

    #         SELECT ticker AS subject_line, date
    #         FROM `xxx.gcdm.campaign_asset_push` a
    #         JOIN `xxx.DASHBOARD_FLAGSHIP.Q6B6_CAMPAIGN_PERFORMANCE_PUSH` p
    #         ON a.HYBRIS_ID = p.HYBRIS_ID
    #     )
        
    #     '''
    #     )

    df_new_sl = client.query_and_wait(QUERY_NEW_SL).to_dataframe()
    df_subject_line_bq = db_dedup.deduplicate(df_new_sl, service_account_json)

    return df_subject_line_bq


def generate_responses_gemini(df_subject_line_bq, model, gemini_success_path, gemini_error_path):
    subject_line_list = df_subject_line_bq['subject_line'].tolist()
    success_lst = [] 
    error_lst = []

    for subject_line in tqdm(subject_line_list): # Show progress bar
        gm.generate_each_sj(subject_line, model, promt_json, success_lst, error_lst, 
                     question_curiosity, question_urgency, question_relavance, 
                     question_value, question_emotion, question_specificity, question_product_feature) #call generate sj function

    gemini_success = pd.json_normalize(success_lst)
    gemini_success.to_parquet(gemini_success_path, index = False)

    gemini_error = pd.json_normalize(error_lst)
    gemini_error.to_parquet(gemini_error_path, index = False)

    return gemini_error


def sl_tagging(gemini_success_path, tagged_sl_path, tagged_raw_sl_path):
    df = pd.read_parquet(gemini_success_path)

    df = df.rename(columns = {
            'subject_line' : 'subject_line',
            'Curiosity.score' : 'curiosity',
            'Curiosity.answer' : 'curiosity_raw',
            'Urgency.score' : 'urgency',
            'Urgency.answer' : 'urgency_raw',
            'Trending Topics/Technologies.score' : 'relevance',
            'Trending Topics/Technologies.answer' : 'relevance_raw',
            'Monetary Value.score' : 'value',
            'Monetary Value.answer' : 'value_raw', 
            'Emotion.score' : 'emotion',
            'Emotion.answer' : 'emotion_raw',
            'Specificity.score' : 'specificity',
            'Specificity.answer' : 'specificity_raw',
            'product_features' : 'product_features'
        }
    )

    df['subject_line'] = df['subject_line'].str.strip()
    df = df.drop_duplicates(subset=['subject_line'])

    df['length_raw'] = df['subject_line'].str.len()
    bins = [0, 30, 35, 40, 50, 60, 1000]
    labels = ['0-30', '30-35', '35-40', '40-50', '50-60', '>60']
    df['length'] = pd.cut(df['length_raw'], bins=bins, labels=labels)

    def extract_emojis(s):
        return ''.join(c for c in s if emoji.is_emoji(c))

    df['emoji_raw'] = df['subject_line'].apply(extract_emojis)
    df['emoji'] = np.where(df['emoji_raw'].str.len() > 0, 1, 0)

    df['exclamation_mark'] = np.where(df['subject_line'].str.endswith('!'), 1, 0)

    df['question_mark'] = np.where(df['subject_line'].str.endswith('?'), 1, 0)

    df['customer_name'] = np.where(df['subject_line'].str.contains('First Name'), 1, 0)

    df['product_features'] = df['product_features'].str.lower()
    df['product_features'] = df['product_features'].str.replace('|'.join(['null', 'none']), '', regex=True)
    df['product_features'] = df['product_features'] + '|' #add | at the end for later concat all campaigns together
    df['product_features'] = np.where(df['product_features'].isna(), '|', df['product_features'])

    cols = ['subject_line', 'curiosity', 'urgency', 'relevance', 'value', 'emotion', 'specificity', 'product_features', 'length', 'emoji', 'exclamation_mark', 'question_mark', 'customer_name']
    cols_raw = ['subject_line', 'curiosity_raw', 'urgency_raw', 'relevance_raw', 'value_raw', 'emotion_raw', 'specificity_raw', 'length_raw', 'emoji_raw']

    df[cols].to_parquet(tagged_sl_path, index=False)
    df[cols_raw].to_parquet(tagged_raw_sl_path, index=False)

    return


def upload_bq(tagged_sl_path, tagged_raw_sl_path, service_account_json):

    df_sj = pd.read_parquet(tagged_sl_path)
    df_sj_raw = pd.read_parquet(tagged_raw_sl_path)

    credentials = service_account.Credentials.from_service_account_file(service_account_json)
    client = bigquery.Client(project = 'xxx', credentials=credentials)
    
    tbl_sj = client.dataset('content').table('subject_line_temp')
    tbl_sj_raw = client.dataset('content').table('subject_line_raw_temp')

    load_job_sj = client.load_table_from_dataframe(df_sj, tbl_sj)
    load_job_sj_raw = client.load_table_from_dataframe(df_sj_raw, tbl_sj_raw)

    load_job_sj.result()
    load_job_sj_raw.result()

    # Merge with existing campaign_asset table
    QUERY = (
        f"""

        MERGE `xxx.content.subject_line` T
        USING `xxx.content.subject_line_temp` S
        ON T.subject_line = S.subject_line
        WHEN NOT MATCHED THEN
        INSERT ROW
        ;

        DROP TABLE IF EXISTS `xxx.content.subject_line_temp`
        ;

        
        MERGE `xxx.content.subject_line_raw` T
        USING `xxx.content.subject_line_raw_temp` S
        ON T.subject_line = S.subject_line
        WHEN NOT MATCHED THEN
        INSERT ROW
        ;

        DROP TABLE IF EXISTS `xxx.content.subject_line_raw_temp`
        """
    )

    query_job = client.query(QUERY)
    results = query_job.result()

    pass