#https://towardsdatascience.com/the-easiest-way-to-interact-with-language-models-4da158cfb5c5
import warnings
import pandas as pd
warnings.filterwarnings("ignore")
import requests
import json
import os
import yaml
import io
import openai
# ----------------------------------------------------------------------------------------------------------------------
from langchain.chains import APIChain
from langchain.chat_models import ChatOpenAI
from langchain.llms import OpenAI, AzureOpenAI
from langchain.requests import RequestsWrapper
from langchain.agents.agent_toolkits.openapi import planner
from langchain.agents.agent_toolkits.openapi.spec import reduce_openapi_spec
from langchain.agents import load_tools, initialize_agent, AgentType, Tool
from langchain.chains.conversation.memory import ConversationBufferWindowMemory
from langchain_experimental.agents.agent_toolkits.pandas.base import create_pandas_dataframe_agent
# ----------------------------------------------------------------------------------------------------------------------
import tools_time_profiler
import tools_DF
# ----------------------------------------------------------------------------------------------------------------------
class Assistant_API(object):
    def __init__(self, filename_config_chat_model,api_spec,access_token=None,chain_type='QA'):
        self.TP = tools_time_profiler.Time_Profiler()
        self.init_cache()
        self.init_model(filename_config_chat_model,chain_type=chain_type)
        self.init_chain(api_spec)
        self.init_agent(api_spec,access_token)
        return
# ----------------------------------------------------------------------------------------------------------------------
    def yaml_to_json(self,text_yaml):
        io_buf = io.StringIO()
        io_buf.write(text_yaml)
        io_buf.seek(0)
        res_json = yaml.load(io_buf, Loader=yaml.Loader)
        return res_json
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

                # openai.api_key = config['azure']['openai_api_key']
                # openai.api_base = config['azure']['openai_api_base']
                # openai.api_version = "2023-05-15"
                model = AzureOpenAI(deployment_name=config['azure']['deployment_name'],openai_api_version=os.environ["OPENAI_API_VERSION"],openai_api_key=os.environ["OPENAI_API_KEY"],openai_api_base=os.environ["OPENAI_API_BASE"])
                #llm = AzureOpenAI(openai_api_type="azure",deployment_name="test1",model_name="gpt-35-turbo")
                #df = pd.DataFrame([])
                #agent = create_pandas_dataframe_agent(model, df, verbose=True)


        self.LLM = model

        return
# ----------------------------------------------------------------------------------------------------------------------
    def get_tools(self, dct_api_spec,access_token):
        def open_api_response(user_query, **kwargs):
            response = openai_agent.run(user_query)
            return response

        headers = {"Authorization": f"Bearer {access_token}","User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)", "Content-Type": "application/json"}
        openai_requests_wrapper = RequestsWrapper(headers=headers)

        openai_agent = planner.create_openapi_agent(reduce_openapi_spec(dct_api_spec),openai_requests_wrapper, self.LLM)
        openai_api_tool = Tool(name="OpenAI API", func=open_api_response,description="""Useful when you need information directly from OpenAI regarding any topic.""")


        return [openai_api_tool]
# ----------------------------------------------------------------------------------------------------------------------
    def get_api_spec(self,api_spec,format='json'):

        if api_spec.find('http')>=0:
            self.api_spec = requests.get(api_spec, verify=False).text
            if format == 'json':
                self.api_spec = self.yaml_to_json(self.api_spec)
        elif api_spec[-5:].find('.json')==0:
            with open(api_spec, 'r') as f:
                self.api_spec = json.dumps(json.load(f))
                self.api_spec = self.yaml_to_json(self.api_spec)
        elif api_spec[-5:].find('.yaml')==0:
            with open(api_spec, 'r') as f:
                self.api_spec = yaml.safe_load(f)
        else:
            self.api_spec = requests.get(api_spec, verify=False).text
            if format == 'json':
                self.api_spec = self.yaml_to_json(self.api_spec)



        return self.api_spec
# ----------------------------------------------------------------------------------------------------------------------
    def init_chain(self,api_spec):
        self.get_api_spec(api_spec)
        #self.chain = get_openapi_chain(api_spec)
        self.chain = APIChain.from_llm_and_api_docs(self.LLM, api_docs=self.api_spec, verbose=True,limit_to_domains=['https://www.example.com'])

        return
# ----------------------------------------------------------------------------------------------------------------------
    def init_agent(self,api_spec,access_token):

        self.get_api_spec(api_spec)
        tools = self.get_tools(self.api_spec,access_token)
        memory = ConversationBufferWindowMemory(memory_key='chat_history',k=5,return_messages=True)
        self.agent = initialize_agent(tools=tools,llm=self.LLM,agent=AgentType.CHAT_ZERO_SHOT_REACT_DESCRIPTION,memory=memory,handle_parsing_errors=True,verbose=True)

        return
# ----------------------------------------------------------------------------------------------------------------------
    def init_cache(self):
        self.filename_cache = './data/output/' + 'cache.json'
        self.dct_cache = {}

        if os.path.isfile(self.filename_cache):
            with open(self.filename_cache, 'r') as f:
                self.dct_cache = json.load(f)
        return
# ----------------------------------------------------------------------------------------------------------------------
    def parse_responce(self,response):
        dct_res = self.yaml_to_json(response)['products']
        df = pd.DataFrame.from_dict(dct_res)
        df_a = pd.DataFrame([])
        for r in range(df.shape[0]):
            attributes = df['attributes'].iloc[r]
            dct_a = dict(zip([a.split(':')[0] for a in attributes], [a.split(':')[-1] for a in attributes]))
            df_a = pd.concat([df_a,pd.DataFrame.from_dict([dct_a])])

        df = pd.concat([df.reset_index().iloc[:,1:], df_a.reset_index().iloc[:,1:]], axis=1)
        df = df[['name','Color','Material','price']]
        response = tools_DF.prettify(df, showindex=False)

        return response
# ----------------------------------------------------------------------------------------------------------------------
    def Q_chain(self,query,texts=None):

        if (texts is None or len(texts) ==0) and (query in self.dct_cache.keys()):
            response = self.dct_cache[query]
        else:
            # response = self.chain.run(query)
            # response = response['response']['products']

            payload_predict = self.chain.api_request_chain.predict(question=query, api_docs=self.api_spec)
            payload_predict = payload_predict.split()[0]
            response = requests.get(payload_predict)
            #response = self.parse_responce(response.text)
            print(payload_predict)

        return response
# ----------------------------------------------------------------------------------------------------------------------
    def Q_agent(self,query,texts=None):

        if (texts is None or len(texts) == 0) and (query in self.dct_cache.keys()):
            response = self.dct_cache[query]
        else:
            response = self.agent.run(query)

        return response
# ----------------------------------------------------------------------------------------------------------------------
