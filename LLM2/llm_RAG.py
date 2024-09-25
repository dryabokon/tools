import numpy
import requests
import warnings
warnings.filterwarnings("ignore")
from halo import Halo
# ----------------------------------------------------------------------------------------------------------------------
from langchain.schema.document import Document
# ----------------------------------------------------------------------------------------------------------------------
from LLM2 import llm_interaction
import tools_time_profiler
# ----------------------------------------------------------------------------------------------------------------------
class RAG(object):
    def __init__(self, chain,Vector_Searcher,do_debug=False,do_spinner=False):
        self.TP = tools_time_profiler.Time_Profiler()
        self.chain = chain
        self.Vector_Searcher = Vector_Searcher
        self.search_mode_hybrid = True
        #print('RAG search mode: ' + ('hybrid' if self.search_mode_hybrid else 'semantic'))

        self.do_debug = do_debug
        self.do_spinner = do_spinner
        return
# ----------------------------------------------------------------------------------------------------------------------
    def texts_to_docs(self,texts):
        docs = [Document(page_content=t, metadata={}) for t in texts]
        return docs
# ----------------------------------------------------------------------------------------------------------------------
    def pretify_string(self,text,N=120):
        lines = []
        line = ""
        for word in text.split():
            if len(line + word) + 1 <= N:
                if line:
                    line += " "
                line += word
            else:
                lines.append(line)
                line = word
        if line:
            lines.append(line)

        result = '\n'.join(lines)

        return result
# ----------------------------------------------------------------------------------------------------------------------
    def init_search_index(self,azure_search_index_name,search_field):
        self.azure_search_index_name =azure_search_index_name
        self.search_field=search_field
        return
# ----------------------------------------------------------------------------------------------------------------------
    def make_rag_prompt(self,query, relevant_passage):
        escaped = relevant_passage.replace("'", "").replace('"', "").replace("\n", " ")
        prompt = ("""You are a helpful and informative bot that answers questions using text from the reference passage included below. \
      Be sure to respond in a complete sentence, being comprehensive, including all relevant background information. \
      However, you are talking to a non-technical audience, so be sure to break down complicated concepts and \
      strike a friendly and converstional tone. \
      If the passage is irrelevant to the answer, you may ignore it.
      QUESTION: '{query}'
      PASSAGE: '{relevant_passage}'

      ANSWER:
      """).format(query=query, relevant_passage=escaped)

        return prompt
# ----------------------------------------------------------------------------------------------------------------------
    def run_query(self, query:str, search_field='token',search_filter_dict=None,limit=4):

        if self.do_spinner:
            self.TP.tic('RAG', verbose=False,reset=True)
            spinner = Halo(spinner={'interval': 100, 'frames': ['⠋', '⠙', '⠹', '⠸', '⠼', '⠴', '⠦', '⠧', '⠇', '⠏']})
            spinner.start()

        #texts = self.Vector_Searcher.search_semantic(self,query,field=search_field,select=select,as_df=False,limit=limit)
        texts = self.Vector_Searcher.search_vector(query, field=search_field, select=self.select, as_df=False,search_filter_dict=search_filter_dict,limit=limit)
        docs = [Document(page_content=t, metadata={}) for t in texts]

        if self.do_spinner:
            spinner.stop()
            spinner.succeed(self.TP.print_duration('RAG', verbose=False))

        texts = [d.page_content for d in docs]

        response = self.chain.run(question=query, input_documents=docs)
        #response = self.chain.invoke({"question":query,"input_documents":docs})['output_text']

        if self.do_debug:
            llm_interaction.display_debug_info(texts)

        return response,texts
    # ----------------------------------------------------------------------------------------------------------------------
    def run_query_v2(self, query:str,field_text='text',field_url='url',search_filter_dict=None,limit=4):
        df = self.Vector_Searcher.search_vector(query, field='token', select=[field_text,field_url], as_df=True,search_filter_dict=search_filter_dict, limit=limit)
        if df.shape[0]>0:
            docs = [Document(page_content=t, metadata={}) for t in df[field_text].tolist()]
            response = self.chain.run(question=query, input_documents=docs)
        else:
            response = 'No relevant documents found.'

        if field_url in df.columns:
            df[field_url] = df[field_url].apply(lambda x: self.Vector_Searcher.generate_signed_url(x, duration_sec=60) if str(x).find('gs:')>=0 else None)
            links = [f"[{1+r}]({df[field_url][r]})" for r in range(df.shape[0])]
            response = response + '\n\n' + ', '.join(links)
        return response
# ----------------------------------------------------------------------------------------------------------------------