import yaml
import pandas as pd
#----------------------------------------------------------------------------------------------------------------------
from azure.core.credentials import AzureKeyCredential
from azure.ai.textanalytics import TextAnalyticsClient, RecognizeLinkedEntitiesAction, RecognizeEntitiesAction
#----------------------------------------------------------------------------------------------------------------------
class Client_NLP(object):
    def __init__(self, filename_config):
        if filename_config is None:
            return
        with open(filename_config, 'r') as config_file:
            config = yaml.safe_load(config_file)
            self.client = TextAnalyticsClient(endpoint=config['azure']['language_processing_service_endpoint'], credential=AzureKeyCredential(config['azure']['language_processing_service_key']))

        return
# ----------------------------------------------------------------------------------------------------------------------
    def split_text(self,text, max_length=5120):

        parts = []
        words = text.split()
        current_part = ""
        for word in words:
            if len(current_part) + len(word) + 1 <= max_length:
                if current_part:
                    current_part += " "
                current_part += word
            else:
                parts.append(current_part)
                current_part = word

        if current_part:
            parts.append(current_part)

        return parts
# ----------------------------------------------------------------------------------------------------------------------
    def deduplicate_entities(self, text, entities):
        E = [e for e in set(entities)]
        df = pd.DataFrame({'E': E, '#': [text.count(e) for e in E]}).sort_values(by=['#'], ascending=False)

        th = 0.01
        critical = df['#'].sum() * th
        idx_is_good = [df['#'].iloc[i:].sum() >= critical for i in range(df.shape[0])]
        res = [e for e in df[idx_is_good]['E']]

        return res
# ----------------------------------------------------------------------------------------------------------------------
    def get_entities(self, long_text,categories=None,min_confidence=0):
        df_res = pd.DataFrame()
        parts = self.split_text(long_text)
        i = 0
        while(i<len(parts)):
            result = self.client.recognize_entities(documents = parts[i:i+5])[0]
            res = [[entity.text,entity.category,entity.subcategory,round(entity.confidence_score, 2),entity.length,entity.offset] for entity in result.entities]
            df = pd.DataFrame(res,columns=['text','category','sub_category','score','length','offset'])

            for category in df['category'].unique():
                if (categories is not None) and (category not in categories):
                    continue
                df_cat = df[df['category']==category]
                for entity in df_cat['text'].unique():
                    df_e = df_cat.loc[df['text']==entity].sort_values(by=['score'],ascending=False).iloc[:1, :]
                    df_res = pd.concat([df_res,df_e])

            i+=5

        df_res = df_res[df_res['score'] >= min_confidence].sort_values(by=['score'], ascending=False)
        res = self.deduplicate_entities(long_text, df_res['text'].values.tolist())

        return res
# ----------------------------------------------------------------------------------------------------------------------
    def analyze_actions(self,long_text):
        parts = self.split_text(long_text,max_length=1000)
        result = self.client.begin_analyze_actions(documents=[parts[0]],actions=[RecognizeEntitiesAction(),RecognizeLinkedEntitiesAction()],show_stats=True)
        results = list(result.result())
        i=0
        return
# ----------------------------------------------------------------------------------------------------------------------