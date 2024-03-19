# ----------------------------------------------------------------------------------------------------------------------
def get_config_openAI():
    class cnfg(object):
        engine = 'openai'
        filename_config_chat_model  = './secrets/private_config_openai.yaml'
        filename_config_emb_model   = './secrets/private_config_openai.yaml'
        #filename_config_vectorstore = './secrets/private_config_pinecone.yaml'
        filename_config_vectorstore = './secrets/GL/private_config_azure_search.yaml'

    return cnfg()
# ----------------------------------------------------------------------------------------------------------------------
def get_config_azure():
    class cnfg(object):
        engine = 'azure'
        filename_config_chat_model = './secrets/GL/private_config_azure_chat.yaml'
        filename_config_emb_model = './secrets/GL/private_config_azure_embeddings.yaml'
        filename_config_vectorstore = './secrets/GL/private_config_azure_search.yaml'
        #filename_config_NLP = './secrets/private_config_azure_NLP.yaml'

    return cnfg()
# ----------------------------------------------------------------------------------------------------------------------
def get_config_neo4j():
    class cnfg(object):
        filename_config_neo4j = './secrets/private_config_neo4j.yaml'

    return cnfg()
# ----------------------------------------------------------------------------------------------------------------------