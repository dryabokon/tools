import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning, module='langchain')
import os
import yaml
from langchain.chat_models import ChatOpenAI,AzureChatOpenAI
from langchain.llms import OpenAI, AzureOpenAI
from langchain_google_vertexai import VertexAI
# ----------------------------------------------------------------------------------------------------------------------
def get_model(filename_config_model, model_type='QA'):


    with open(filename_config_model) as config_file:
        config = yaml.safe_load(config_file)
        if 'openai' in config:
            #engine = "gpt-3.5-turbo"
            #engine = "gpt-4"
            #engine = "gpt-4-1106-preview"
            engine = "gpt-4o"
            openai_api_key = config['openai']['key']
            os.environ["OPENAI_API_KEY"] = openai_api_key

            if model_type == 'QA':
                model = ChatOpenAI(temperature=0, openai_api_key=openai_api_key,model_name=engine)
            else:
                model = OpenAI(temperature=0, openai_api_key=openai_api_key,model_name=engine)

            print(f'OpenAI {model.model_name} initialized')

        elif 'azure' in config:
            import openai
            os.environ["OPENAI_API_TYPE"] = "azure"
            os.environ["OPENAI_API_VERSION"] = config['azure']['openai_api_version']
            os.environ["OPENAI_API_BASE"] = config['azure']['openai_api_base']
            os.environ["OPENAI_API_KEY"] = config['azure']['openai_api_key']
            openai.api_key = config['azure']['openai_api_key']

            openai.api_type = "azure"
            openai.api_version = config['azure']['openai_api_version']
            openai.api_base = config['azure']['openai_api_base']
            openai.api_key = config['azure']['openai_api_key']

            if model_type== 'QA' :
                model = AzureChatOpenAI(deployment_name=config['azure']['deployment_name'])
            else:
                model = AzureOpenAI(deployment_name=config['azure']['deployment_name'])

            print(f'Azure {model.model_name} initialized')
        elif 'GCP' in config:
            model = VertexAI(model_name="gemini-pro")
            #model = ChatGoogleGenerativeAI(model="gemini-pro",google_api_key=config['GCP']['google_api_key'])
            print(f'GCP {model.model_name} initialized')

        return model
# ----------------------------------------------------------------------------------------------------------------------