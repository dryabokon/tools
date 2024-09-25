#https://neo4j.com/docs/graph-data-science/current/machine-learning/node-property-prediction/nodeclassification-pipelines/config/
# --------------------------------------------------------------------------------------------------------------------
import pandas as pd
import numpy
import os
import yaml
import json
import neo4j
import uuid
import inspect
# --------------------------------------------------------------------------------------------------------------------
from graphdatascience import GraphDataScience
import tools_SSH
import tools_DF
import tools_time_profiler
import tools_Logger
# --------------------------------------------------------------------------------------------------------------------
class processor_Neo4j(object):
    def __init__(self,filename_config_neo4j,filename_config_ssh,folder_out):
        self.load_private_config(filename_config_neo4j)
        self.folder_out = folder_out
        self.SSH = tools_SSH.SSH_Client(filename_config_ssh)
        self.TP = tools_time_profiler.Time_Profiler()
        self.L = tools_Logger.Logger(self.folder_out+'log_Neo4j.txt')
        self.verbose = False
        return
# ----------------------------------------------------------------------------------------------------------------------
    def load_private_config(self,filename_in):

        if filename_in is None:return None
        if not os.path.isfile(filename_in):return None

        with open(filename_in, 'r') as config_file:
            config = yaml.safe_load(config_file)
            url = f"bolt://{config['database']['host']}:{config['database']['port']}"
            auth = (config['database']['user'], config['database']['password'])
            self.database = config['database']['dbname']
            self.driver = neo4j.GraphDatabase.driver(url, auth=auth)
            #self.gds = GraphDataScience(url, auth=auth, database=self.database)


        return
# ----------------------------------------------------------------------------------------------------------------------
    def close(self):
        if self.driver:
            self.driver.close()
        return
# ----------------------------------------------------------------------------------------------------------------------
    def execute_query(self,query):
        self.L.write(query)

        if self.verbose:
            print(query + ';\n')

        df = self.driver.execute_query(query)
        # with self.driver.session() as session:
        #     result = session.run(query)

        return
# ----------------------------------------------------------------------------------------------------------------------
    def df_to_json(self,df,filename_json):
        df=df.dropna()
        columns = [c for c in df.columns]

        with open(self.folder_out + filename_json, 'w+', encoding='utf-8') as f:
            for r in range(df.shape[0]):
                dct_record = dict(zip(columns, [(df[key].iloc[r]) for key in columns]))
                json.dump(dct_record, f, indent=' ',default=int)

        return
# ----------------------------------------------------------------------------------------------------------------------
    def export_json_to_neo4j(self, filename_json, dct_entities,dct_relations,drop_if_exists):

        self.SSH.scp_file_to_remote(self.folder_out + filename_json, '~/' + filename_json)
        self.SSH.execure_command(f'sudo docker cp ~/{filename_json} "neo4j:/var/lib/neo4j/data/{filename_json}"')
        self.SSH.execure_command(f'sudo rm ~/{filename_json}')


        entities = [k for k in dct_entities.keys()]

        if drop_if_exists:
            self.execute_query('MATCH (n) DETACH DELETE n')

        q2 =  f'CALL apoc.load.json("./data/{filename_json}") YIELD value AS value \n'

        for e_id, (entity,properties) in enumerate(dct_entities.items()):
            q2 += 'MERGE (e%d:%s {%s})\n'%(e_id,entity,','.join(['%s:value.%s'%(p,p) for p in properties]))

        for relation_name,list_entities in dct_relations.items():
            q2 += 'MERGE (e%d)-[:%s]->(e%d)'%(entities.index(list_entities[0]),relation_name,entities.index(list_entities[1]))

        for q in [q2]:
            self.execute_query(q)

        self.SSH.execure_command(f'sudo docker exec neo4j rm "/var/lib/neo4j/data/{filename_json}"')

        return
# ----------------------------------------------------------------------------------------------------------------------
    def export_df_to_neo4j(self,df,dct_entities, dct_relations,drop_if_exists):
        self.TP.tic(inspect.currentframe().f_code.co_name, reset=True)
        filename_tmp = uuid.uuid4().hex + '.json'
        self.df_to_json(df, filename_tmp)
        self.export_json_to_neo4j(filename_tmp, dct_entities, dct_relations,drop_if_exists)
        os.remove(self.folder_out + filename_tmp)
        self.TP.print_duration(inspect.currentframe().f_code.co_name)
        return
# ----------------------------------------------------------------------------------------------------------------------
    def import_from_neo4j(self,query,filename_out):

        filename_tmp = uuid.uuid4().hex + '.json'
        self.execute_query('CALL apoc.export.json.query("%s","./data/%s",{})' % (query,filename_tmp))
        self.transfer(filename_tmp)

        with open(self.folder_out + filename_tmp, 'r') as f:
            cols = [k.split('.')[-1] for k in json.loads(f.readline()).keys()]
            df = pd.DataFrame([[k for k in json.loads(line).values()] for line in f.readlines()],columns=cols)

        df.to_csv(self.folder_out+filename_out,index=False)
        os.remove(self.folder_out + filename_tmp)

        return
# ----------------------------------------------------------------------------------------------------------------------
    def create_graph_native(self,dct_entities,dct_relations):

        graph_name = 'xxx'

        q3 = 'CALL gds.graph.drop("%s",false) YIELD graphName'%graph_name
        rows = [entity + ': {properties: [%s]}' % (','.join(['\'%s\'' % p for p in properties])) for e_id, (entity, properties) in enumerate(dct_entities.items())]
        str_prop = '{%s}' % (',\n'.join(rows))
        rows2 = [relation_name + ': {orientation: \'UNDIRECTED\'}' for relation_name in dct_relations.keys()]
        str_orient = '{%s}' % (',\n'.join(rows2))
        #str_orient = '\'*\''
        q4 = 'CALL gds.graph.project("%s",%s,\n%s)' % (graph_name,str_prop, str_orient)

        for q in [q3,q4]:
            self.execute_query(q)

        return graph_name
# ----------------------------------------------------------------------------------------------------------------------
    def create_graph_gds(self,dct_entities,dct_relations):

        self.execute_query('CALL gds.graph.drop("xxx",false)')
        dct1 = {}
        for entity, properties in dct_entities.items():
            dct1[entity]={'properties':properties}

        dct2 = {}
        for relation_name in dct_relations.keys():
            dct2[relation_name]={"orientation": "UNDIRECTED", "aggregation": "SINGLE"}

        G, _ = self.gds.graph.project("xxx",dct1,dct2)

        return G
# ----------------------------------------------------------------------------------------------------------------------
    def train_graphSage_encoder_native(self, graph_name,dct_entities, dct_relations, features, emb_dims=2):
        self.TP.tic(inspect.currentframe().f_code.co_name, reset=True)
        model_name = 'multiLabelModel'

        quoted_features = (','.join([f'"%s"'%feature for feature in features]))
        quoted_entities = (','.join(['\'' + e + '\'' for e in dct_entities.keys()]))
        quoted_relationship = (','.join(['\'' + e + '\'' for e in dct_relations.keys()]))

        q5 = 'CALL gds.beta.model.drop("%s",false)'%model_name
        #q6 = 'CALL gds.beta.graphSage.train(\'%s\',{modelName: \'%s\', nodeLabels:[%s], relationshipTypes:[%s],epochs:2000, learningRate:0.0001,projectedFeatureDimension:%d, embeddingDimension: %d,featureProperties: [%s]})'%(graph_name,model_name,quoted_entities,quoted_relationship,emb_dims,emb_dims,quoted_features)
        q6 = 'CALL gds.beta.graphSage.train(\'%s\',{modelName: \'%s\', epochs:2000, learningRate:0.0001,projectedFeatureDimension:%d, embeddingDimension: %d,featureProperties: [%s]})'%(graph_name,model_name,emb_dims,emb_dims,quoted_features)

        for q in [q5,q6]:
            self.execute_query(q)

        self.TP.print_duration(inspect.currentframe().f_code.co_name)
        return model_name
# ----------------------------------------------------------------------------------------------------------------------
    def train_model_classification_native(self,graph_name,entity, target_property,features):
        self.TP.tic(inspect.currentframe().f_code.co_name, reset=True)
        quoted_features = (','.join([f'"%s"' % feature for feature in features]))
        model_name = 'xxx_model'

        q5 = 'CALL gds.beta.model.drop("%s",false)'%model_name
        q6 = 'CALL gds.beta.pipeline.drop("pipe",false)'
        q7 = 'CALL gds.beta.pipeline.nodeClassification.create("pipe")'
        q9 = "CALL gds.beta.pipeline.nodeClassification.selectFeatures('pipe', [%s])"%quoted_features
        q10= "CALL gds.beta.pipeline.nodeClassification.configureSplit('pipe', {testFraction: 0.5,validationFolds: 5})"
        #q11 = "CALL gds.beta.pipeline.nodeClassification.addLogisticRegression('pipe', {maxEpochs: 500, penalty: {range: [1e-4, 1e2]}})"
        q11 = "CALL gds.beta.pipeline.nodeClassification.addRandomForest('pipe', {maxDepth: 5})"

        q12 = "CALL gds.alpha.pipeline.nodeClassification.configureAutoTuning('pipe', {maxTrials: 2})"
        q13 = "CALL gds.beta.pipeline.nodeClassification.train('%s', {pipeline: 'pipe', modelName: '%s', targetNodeLabels: ['%s'], targetProperty: '%s', metrics: ['F1_WEIGHTED']})"%(graph_name,model_name,entity,target_property)

        for q in [q5,q6,q7,q9,q10,q11,q12,q13]:
            self.execute_query(q)

        self.TP.print_duration(inspect.currentframe().f_code.co_name)
        return model_name
# ----------------------------------------------------------------------------------------------------------------------
    def train_model_classification_gds(self, G, entity, target_property, features):
        self.TP.tic(inspect.currentframe().f_code.co_name, reset=True)
        self.execute_query('CALL gds.beta.pipeline.drop("xxx_pipeline",false)')
        self.execute_query('CALL gds.beta.model.drop("xxx_model",false)')
        node_pipeline, _ = self.gds.beta.pipeline.nodeClassification.create("xxx_pipeline")

        node_pipeline.configureSplit(testFraction=0.5, validationFolds=5)
        #node_pipeline.addLogisticRegression(maxEpochs=10000, penalty=(0.0, 0.5))
        node_pipeline.addRandomForest(maxDepth=5)
        node_pipeline.configureAutoTuning(maxTrials=5)

        node_pipeline.selectFeatures(features)
        model, stats = node_pipeline.train(G, targetNodeLabels=[entity], targetProperty=target_property,modelName="xxx_model", metrics=["F1_WEIGHTED"])
        print('F1_WEIGHTED train',stats["modelInfo"]["metrics"]["F1_WEIGHTED"]["train"]['avg'])
        print('F1_WEIGHTED test', stats["modelInfo"]["metrics"]["F1_WEIGHTED"]["test"])
        self.TP.print_duration(inspect.currentframe().f_code.co_name)
        return model
# ----------------------------------------------------------------------------------------------------------------------
    def transfer(self,filename):
        self.SSH.execure_command('sudo docker cp "neo4j:/var/lib/neo4j/data/%s" ~/%s'%(filename,filename))
        self.SSH.scp_file_from_remote('~/%s'%filename, self.folder_out + '%s'%filename)
        self.SSH.execure_command('sudo rm ~/%s'%filename)
        self.SSH.execure_command(f'sudo docker exec neo4j rm /var/lib/neo4j/data/%s'%filename)
        return
# ----------------------------------------------------------------------------------------------------------------------
    def run_graphSage_encoder_native(self, graph_name, model_name, entity, identity, dct_entities, dct_relations):
        self.TP.tic(inspect.currentframe().f_code.co_name, reset=True)
        quoted_entities = (','.join(['\'' + e + '\'' for e in dct_entities.keys()]))
        quoted_relationship = (','.join(['\'' + e + '\'' for e in dct_relations.keys()]))

        #q_stream = 'CALL gds.beta.graphSage.stream(\'%s\',{modelName: \'%s\',nodeLabels:[%s], relationshipTypes:[%s]}) YIELD nodeId, embedding' % (graph_name, model_name, quoted_entities, quoted_relationship)
        q_stream = 'CALL gds.beta.graphSage.stream(\'%s\',{modelName: \'%s\'}) YIELD nodeId, embedding' % (graph_name, model_name)
        q8 = 'CALL apoc.export.json.query("%s","./data/nodeId_embedding.json",{})' % q_stream
        q9 = 'CALL apoc.export.json.query("match (e:%s) return ID(e), e.%s","./data/nodeId_identity.json",{})' % (entity, identity)

        for q in [q8, q9]:
            self.execute_query(q)

        self.transfer('nodeId_embedding.json')
        self.transfer('nodeId_identity.json')

        self.TP.print_duration(inspect.currentframe().f_code.co_name)
        return
# ----------------------------------------------------------------------------------------------------------------------
    def run_fastRP_encoder_native(self, graph_name, entity, identity, emb_dims=64):
        self.TP.tic(inspect.currentframe().f_code.co_name, reset=True)
        #q_stream = 'CALL gds.fastRP.stream(\'%s\',{nodeLabels:[%s], featureProperties: [%s], embeddingDimension: %d}) YIELD nodeId, embedding' %(graph_name,quoted_entities,quoted_features,emb_dims)
        #q_stream = 'CALL gds.fastRP.stream(\'%s\',{nodeLabels:[\'%s\'], embeddingDimension: %d}) YIELD nodeId, embedding' % (graph_name,entity,emb_dims)
        q_stream = 'CALL gds.fastRP.stream(\'%s\',{embeddingDimension: %d}) YIELD nodeId, embedding' % (graph_name,emb_dims)

        q8 = 'CALL apoc.export.json.query("%s","./data/nodeId_embedding.json",{})' % q_stream
        q9 = 'CALL apoc.export.json.query("match (e:%s) return ID(e), e.%s","./data/nodeId_identity.json",{})' % (entity, identity)

        for q in [q8, q9]:
            self.execute_query(q)

        self.transfer('nodeId_embedding.json')
        self.transfer('nodeId_identity.json')
        self.TP.print_duration(inspect.currentframe().f_code.co_name)
        return
# ----------------------------------------------------------------------------------------------------------------------
    def run_model_classification_gds(self, model, G, entity, identity):
        self.TP.tic(inspect.currentframe().f_code.co_name, reset=True)
        df_predict = model.predict_stream(G, targetNodeLabels=[entity], modelName="xxx_model",includePredictedProbabilities=True)
        df_predict.to_csv(self.folder_out + 'nodeId_embedding.csv', index=False, float_format='%.2f')

        q9 = 'CALL apoc.export.json.query("match (e:%s) return ID(e), e.%s","./data/nodeId_identity.json",{})'%(entity,identity)
        for q in [q9]:
            self.execute_query(q)

        self.transfer('nodeId_identity.json')

        self.TP.print_duration(inspect.currentframe().f_code.co_name)
        return
# ----------------------------------------------------------------------------------------------------------------------
    def run_model_classification_native(self,graph_name,model_name,entity,identity):
        self.TP.tic(inspect.currentframe().f_code.co_name, reset=True)
        q_stream = 'CALL gds.beta.pipeline.nodeClassification.predict.stream(\'%s\',{modelName: \'%s\', targetNodeLabels: [\'%s\'], includePredictedProbabilities: true}) YIELD nodeId, predictedClass, predictedProbabilities'%(graph_name,model_name,entity)
        q8 = 'CALL apoc.export.json.query("%s","./data/nodeId_embedding.json",{})'%(q_stream)
        q9 = 'CALL apoc.export.json.query("match (e:%s) return ID(e), e.%s","./data/nodeId_identity.json",{})' % (entity, identity)

        for q in [q8,q9]:
            self.execute_query(q)

        self.transfer('nodeId_embedding.json')
        self.transfer('nodeId_identity.json')

        self.TP.print_duration(inspect.currentframe().f_code.co_name)
        return
# ----------------------------------------------------------------------------------------------------------------------
    def fetch_output_from_neo4j(self, identity, filename_out=None):

        with open(self.folder_out + './nodeId_identity.json', 'r') as f:
            df_nodeId_identity = pd.DataFrame([[k for k in json.loads(line).values()] for line in f.readlines()],columns=['nodeId',identity])

        if os.path.isfile(self.folder_out + './nodeId_embedding.json'):
            with open(self.folder_out + './nodeId_embedding.json', 'r') as f:
                df_nodeId_embedding   = pd.DataFrame([[k for k in json.loads(line).values()] for line in f.readlines()])
        else:
            df_nodeId_embedding = pd.read_csv(self.folder_out + './nodeId_embedding.csv')

        df_nodeId_embedding = df_nodeId_embedding.rename(columns={df_nodeId_embedding.columns[0]:'nodeId'})
        df_nodeId_embedding = tools_DF.fetch(df_nodeId_embedding,'nodeId',df_nodeId_identity,'nodeId',identity)
        df_nodeId_embedding = df_nodeId_embedding[~df_nodeId_embedding[identity].isna()].drop(columns=['nodeId'])
        df_nodeId_embedding = df_nodeId_embedding.iloc[:,numpy.roll(numpy.arange(0,df_nodeId_embedding.shape[1]), 1)]
        df_nodeId_embedding[identity] = df_nodeId_embedding[identity].astype(int)
        df_nodeId_embedding = tools_DF.auto_explode(df_nodeId_embedding, df_nodeId_embedding.columns[-1])
        if filename_out is not None:
            df_nodeId_embedding.to_csv(self.folder_out + filename_out, index=False, float_format='%.2f')

        return df_nodeId_embedding
# ----------------------------------------------------------------------------------------------------------------------

