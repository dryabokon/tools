import warnings
import pandas as pd
warnings.filterwarnings("ignore")
import os
import yaml
# ----------------------------------------------------------------------------------------------------------------------
from langchain.chat_models import ChatOpenAI
from langchain.llms import OpenAI, AzureOpenAI
from langchain.agents import AgentType
from langchain_experimental.agents.agent_toolkits.pandas.base import create_pandas_dataframe_agent

# ----------------------------------------------------------------------------------------------------------------------
import tools_time_profiler
import tools_DF
# ----------------------------------------------------------------------------------------------------------------------
class Assistant_Pandas(object):
    def __init__(self, filename_config_chat_model,df,chain_type='QA'):
        self.TP = tools_time_profiler.Time_Profiler()
        self.init_model(filename_config_chat_model,chain_type=chain_type)
        self.init_agent(df)
        return
# ----------------------------------------------------------------------------------------------------------------------
    def init_model(self,filename_config_chat_model,chain_type='QA'):
        with open(filename_config_chat_model, 'r') as config_file:
            config = yaml.safe_load(config_file)
            if 'openai' in config:
                openai_api_key = config['openai']['key']
                os.environ["OPENAI_API_KEY"] = openai_api_key
                if chain_type == 'QA':
                    model = ChatOpenAI(temperature=0, openai_api_key=openai_api_key)
                else:
                    model = OpenAI(temperature=0, openai_api_key=openai_api_key)

            elif 'azure' in config:
                os.environ["OPENAI_API_TYPE"] = "azure"
                os.environ["OPENAI_API_VERSION"] = "2023-05-15"
                os.environ["OPENAI_API_BASE"] = config['azure']['openai_api_base']
                os.environ["OPENAI_API_KEY"] = config['azure']['openai_api_key']
                model = AzureOpenAI(deployment_name=config['azure']['deployment_name'],openai_api_version=os.environ["OPENAI_API_VERSION"],openai_api_key=os.environ["OPENAI_API_KEY"],openai_api_base=os.environ["OPENAI_API_BASE"])

        self.LLM = model

        return
# ----------------------------------------------------------------------------------------------------------------------
    def init_agent(self,df):
        self.agent = create_pandas_dataframe_agent(self.LLM,df,verbose=True,agent_type=AgentType.OPENAI_FUNCTIONS,handle_parsing_errors=True)
        return
# ----------------------------------------------------------------------------------------------------------------------
    def Q_agent(self,query):
        response = self.agent.run(query)
        return response
# ----------------------------------------------------------------------------------------------------------------------
