import pandas as pd
from google.cloud import bigquery
from langchain_community.document_loaders import BigQueryLoader
# ----------------------------------------------------------------------------------------------------------------------
from LLM2 import llm_models,llm_config
# ----------------------------------------------------------------------------------------------------------------------
class Agent_BQ(object):
    def __init__(self,project, dataset,table_name):
        self.LLM = llm_models.get_model(llm_config.get_config_GCP().filename_config_chat_model, model_type='QA')
        self.table_ddl = self.get_table_ddl(project, dataset, table_name)
        desc = {'Speed': 'Speed of the train',
                'AveTempC': 'Average Temperature in Celsius',
                'UTCTimeDate': 'Time and Date in UTC',
                'Unit': 'Unit of train',
                'wheel': 'Wheel of the train',
                'WHI': 'wheel health index',
                'BHI': 'business health index'}


        self.desc = ' '.join([f'{k} is {v}. ' for k, v in desc.items()])
        self.history = []
        self.client = bigquery.Client(project=project, location='US')

        return
# ----------------------------------------------------------------------------------------------------------------------
    def get_table_ddl(self,project, dataset, table):
        query = f""" SELECT table_name, ddl FROM `{dataset}.INFORMATION_SCHEMA.TABLES` WHERE table_name = '{table}'; """
        loader = BigQueryLoader(query, project=project, metadata_columns="table_name", page_content_columns="ddl")
        data = loader.load()
        res = data[0].page_content
        return res
# ----------------------------------------------------------------------------------------------------------------------
    def create_prompt(self,query):
        hst = '\n'.join(self.history[:3])
        prompt = (f'This is the table dll {self.table_ddl}. '
                  #f'This is description of columns {self.desc}. '
                  f'This is the history of previous questions and answers: {hst}.'
                  f'Write a BigQuery query to answers the following question: {query}. '
                  f'Tag your reply with START and END signatures.'
                  f'Ensure content between START and END signatures can be executed as-is on BigQuery!')
        return prompt
# ----------------------------------------------------------------------------------------------------------------------
    def ex_simple_query(self,SQL):
        from sqlalchemy import create_engine
        #SQL = f'SELECT * FROM {self.table_name} limit 10'

        uri = f"bigquery://{self.project}/{self.dataset}"
        db = create_engine(uri)
        with db.connect() as conn:
            df = pd.read_sql(SQL, conn)

        return df
# ----------------------------------------------------------------------------------------------------------------------
    def run_query(self, query):
        prompt = self.create_prompt(query)
        SQL_query = self.LLM.predict(prompt)
        SQL_query = SQL_query[SQL_query.find('START')+5:SQL_query.find('END')]
        SQL_query = SQL_query.replace('```sql', '').replace('`', '')

        self.history.append('Question: ' + query + '\n' + '. Result:' + SQL_query + '\n')
        #print(SQL_query)
        try:
            df = self.client.query(SQL_query).result().to_dataframe()
        except Exception as e:
            df = pd.DataFrame({})

        return df, [SQL_query]
# ----------------------------------------------------------------------------------------------------------------------
