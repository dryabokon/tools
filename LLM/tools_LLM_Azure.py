import yaml
import os
from langchain.llms import AzureOpenAI
from langchain.chat_models import AzureChatOpenAI
#----------------------------------------------------------------------------------------------------------------------
def LLM(filename_config,chatmode=False):
    with open(filename_config, 'r') as config_file:
        config = yaml.safe_load(config_file)
        os.environ["OPENAI_API_TYPE"] = "azure"
        os.environ["OPENAI_API_VERSION"] = "2023-05-15"
        #os.environ["OPENAI_API_BASE"] = config['azure']['openai_api_base']
        os.environ["OPENAI_API_KEY"] = config['azure']['openai_api_key']

    if chatmode:
        LLM = AzureChatOpenAI(deployment_name=config['azure']['deployment_name'],
                              openai_api_version=os.environ["OPENAI_API_VERSION"],
                              openai_api_key=config['azure']['openai_api_key'],
                              openai_api_base=config['azure']['openai_api_base'])
    else:
        LLM = AzureOpenAI(deployment_name=config['azure']['deployment_name'])

    return LLM
#----------------------------------------------------------------------------------------------------------------------