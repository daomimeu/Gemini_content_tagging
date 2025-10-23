import pandas as pd
import numpy as np

from google.oauth2 import service_account
from google.cloud import bigquery


def deduplicate(df, service_account_json):
    data = df[['subject_line']]

    credentials = service_account.Credentials.from_service_account_file(service_account_json)
    client_digitas = bigquery.Client(project = 'seaocdm-data-digitas', credentials=credentials)

    QUERY_ALL_SL = (
        '''
        SELECT DISTINCT subject_line AS db_subject_line
        FROM `seaocdm-data-digitas.content.subject_line`

        '''
    )

    df_all_sl = client_digitas.query_and_wait(QUERY_ALL_SL).to_dataframe()

    data = data.merge(df_all_sl, how='left', left_on='subject_line', right_on='db_subject_line')
    data = data[data['db_subject_line'].isna()]

    return data[['subject_line']]