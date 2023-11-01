import warnings
warnings.filterwarnings("ignore")
import requests
import json
import os
import yaml
import io
# ----------------------------------------------------------------------------------------------------------------------
from langchain.chains.openai_functions.openapi import get_openapi_chain
from langchain.requests import RequestsWrapper
from langchain.chat_models import ChatOpenAI,AzureChatOpenAI
from langchain.llms import OpenAI, AzureOpenAI
from langchain.agents.agent_toolkits.openapi import planner
from langchain.agents.agent_toolkits.openapi.spec import reduce_openapi_spec
from langchain.agents import load_tools, initialize_agent, AgentType, Tool
from langchain.agents.agent_toolkits import NLAToolkit
from langchain.chains.conversation.memory import ConversationBufferWindowMemory
# ----------------------------------------------------------------------------------------------------------------------
import tools_time_profiler
# ----------------------------------------------------------------------------------------------------------------------
class Agent(object):
    def __init__(self, filename_config_chat_model,chain_type='QA'):
        self.TP = tools_time_profiler.Time_Profiler()
        self.init_cache()
        self.init_model(filename_config_chat_model, chain_type=chain_type)
        self.init_agent_v0()
        #self.init_agent_v1()
        return
# ----------------------------------------------------------------------------------------------------------------------
    def yaml_to_json(self,text_yaml):
        io_buf = io.StringIO()
        io_buf.write(text_yaml)
        io_buf.seek(0)
        res_json = yaml.load(io_buf, Loader=yaml.Loader)
        return res_json
# ----------------------------------------------------------------------------------------------------------------------
    def get_openai_api_spec(self,URL_yaml_file,format='json'):
        openai_api_spec = requests.get(URL_yaml_file, verify=False).text
        if format == 'json':
            openai_api_spec = self.yaml_to_json(openai_api_spec)
        return openai_api_spec
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
                os.environ["OPENAI_API_TYPE"] = "azure"
                os.environ["OPENAI_API_VERSION"] = "2023-05-15"
                os.environ["OPENAI_API_BASE"] = config['azure']['openai_api_base']
                os.environ["OPENAI_API_KEY"] = config['azure']['openai_api_key']
                if chain_type == 'QA':
                    model = AzureChatOpenAI(deployment_name=config['azure']['deployment_name'], openai_api_version=os.environ["OPENAI_API_VERSION"],openai_api_key=os.environ["OPENAI_API_KEY"],openai_api_base=os.environ["OPENAI_API_BASE"])
                else:
                    model = AzureOpenAI(deployment_name=config['azure']['deployment_name'], openai_api_version=os.environ["OPENAI_API_VERSION"],openai_api_key=os.environ["OPENAI_API_KEY"],openai_api_base=os.environ["OPENAI_API_BASE"])

        self.LLM = model
        return
# ----------------------------------------------------------------------------------------------------------------------
    def get_tools(self,openai_api_spec_json):
        def open_api_response(user_query, **kwargs):
            response = openai_agent.run(user_query)
            return response

        openai_requests_wrapper = RequestsWrapper(headers={"Authorization": f"Bearer {os.getenv('OPENAI_API_KEY')}"})

        openai_agent = planner.create_openapi_agent(reduce_openapi_spec(openai_api_spec_json), openai_requests_wrapper, self.LLM)
        openai_api_tool = Tool(name="OpenAI API",func=open_api_response,description="""Useful when you need information directly from OpenAI regarding any topic.""")

        available_tools = {'Math': "llm-math",'Open AI API Tool': openai_api_tool}
        tools =  load_tools([available_tools[tool] for tool in available_tools.keys() if isinstance(available_tools[tool], str) == True], llm=self.LLM)
        tools += [          available_tools[tool] for tool in available_tools.keys() if isinstance(available_tools[tool], str) != True]
        return tools
# ----------------------------------------------------------------------------------------------------------------------
    def init_agent_v0(self):

        tools = self.get_tools(self.get_openai_api_spec("https://raw.githubusercontent.com/openai/openai-openapi/master/openapi.yaml"))
        memory = ConversationBufferWindowMemory(memory_key='chat_history',k=5,return_messages=True)
        self.agent = initialize_agent(tools=tools,llm=self.LLM,agent=AgentType.CHAT_ZERO_SHOT_REACT_DESCRIPTION,memory=memory,handle_parsing_errors=True,verbose=True)

        return
# ----------------------------------------------------------------------------------------------------------------------
    def init_agent_v1(self):

        tools_NLA = []

        # tools_NLA.extend(NLAToolkit.from_llm_and_url(self.LLM,"https://www.transvribe.com/ai-plugin/openapi.yaml").get_tools())  # Undestands YouTube videos
        # tools_NLA.extend(NLAToolkit.from_llm_and_url(self.LLM,"https://chatgpt-plugin-ts.transitive-bullshit.workers.dev/openapi.json").get_tools())  # Generates ASII art
        # tools_NLA.extend(NLAToolkit.from_llm_and_url(self.LLM,"https://www.greenyroad.com/openapi.yaml").get_tools())  # Reads URLs and provides context
        # tools_NLA.extend(NLAToolkit.from_llm_and_url(self.LLM,"https://api.schooldigger.com/swagger/docs/v2.0").get_tools())  # Information about school districts
        #tools_NLA.extend(NLAToolkit.from_llm_and_url(self.LLM,"https://api.tasty.co/.well-known/openapi.yaml").get_tools()),  # Recipes
        # tools_NLA.extend(NLAToolkit.from_llm_and_url(self.LLM,"https://www.wolframalpha.com/.well-known/apispec.json").get_tools())  # Wolfram Alpha
        #tools_NLA.extend(NLAToolkit.from_llm_and_url(self.LLM,"https://www.freetv-app.com/openapi.json").get_tools())  # Latest news (works)
        # tools_NLA.extend(NLAToolkit.from_llm_and_url(self.LLM,"https://chatgpt-plugin-dexa-lex-fridman.transitive-bullshit.workers.dev/openapi.json").get_tools())  # Searches Lex Friedman's podcast
        # tools_NLA.extend(NLAToolkit.from_llm_and_url(self.LLM,"https://websearch.plugsugar.com/api/openapi_yaml").get_tools())  # Search web (works)
        # tools_NLA.extend(NLAToolkit.from_llm_and_url(self.LLM,"https://api.speak.com/openapi.yaml").get_tools())
        tools_NLA.extend(NLAToolkit.from_llm_and_url(self.LLM,"https://www.klarna.com/us/shopping/public/openai/v0/api-docs/").get_tools())  # shop online
        #tools_NLA.extend(NLAToolkit.from_llm_and_url(self.LLM,"https://server.shop.app/openai/v1/api.json").get_tools())  # Search for products
        # agent_kwargs={"format_instructions": openapi_format_instructions}
        self.agent = initialize_agent(tools_NLA, self.LLM, agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION, verbose=True)

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
    def Q(self,query,texts=None):

        if (texts is None or len(texts) ==0) and (query in self.dct_cache.keys()):
            responce = self.dct_cache[query]
        else:
            responce = self.agent(query)
            #responce = self.agent.run(query)

        return responce
# ----------------------------------------------------------------------------------------------------------------------

