import os
import warnings
warnings.filterwarnings( 'ignore', module = 'langchain' )
# ----------------------------------------------------------------------------------------------------------------------
import re
import numpy
import numpy_financial
from langchain.chains import LLMMathChain
from langchain.agents import Tool
from langchain.agents.agent_toolkits import NLAToolkit
from langchain_experimental.tools.python.tool import PythonAstREPLTool
from langchain.tools import StructuredTool
# ----------------------------------------------------------------------------------------------------------------------
from LLM2 import llm_config,llm_models,llm_chains,llm_RAG,llm_interaction,llm_tools,llm_Agent
# ----------------------------------------------------------------------------------------------------------------------
llm_cnfg = llm_config.get_config_azure()
#llm_cnfg = llm_config.get_config_openAI()
# ----------------------------------------------------------------------------------------------------------------------

def custom_func_IRR_calc(cash_flows:str):
    #print('custom_func_IRR_calc executed..')
    A = [re.sub("[^0-9-+.]", "", x) for x in cash_flows.split()]
    A = [a for a in A if len(a) > 0]
    A = numpy.array(A).astype(float)
    irr = numpy_financial.irr(A)
    return irr
# ----------------------------------------------------------------------------------------------------------------------
def custom_func_sales_for_target_irr(cash_flows:list, target_irr: float) -> float:
    from scipy.optimize import fsolve
    #print('custom_func_sales_for_target_irr executed..')
    def f_irr(x, *data):
        cash_flows = data[:-1]
        target_irr = data[-1]
        A = [c for c in cash_flows]
        A.extend(x)
        delta = numpy_financial.irr(A) - target_irr
        return delta

    data = [c for c in cash_flows]
    data.extend([target_irr])
    x = fsolve(f_irr, 0, args=tuple(data))[0]
    data[-1] = x
    check = numpy_financial.irr(data)

    # print(cash_flows)
    # print('target IRR: ',target_irr)
    # print('sale: ',x)
    # print('check: ',check)

    return float(x)
# ----------------------------------------------------------------------------------------------------------------------
def custom_func_sales_for_target_irr_single(target_and_cash_flows:str):
    #print('custom_func_sales_for_target_irr executed..')
    def f_irr(x, *data):
        cash_flows = data[:-1]
        target_irr = data[-1]
        A = [c for c in cash_flows]
        A.extend(x)
        delta = numpy_financial.irr(A) - target_irr
        return delta

    A = [re.sub("[^0-9-+.]", "", x) for x in target_and_cash_flows.split()]
    A = [a for a in A if len(a) > 0]
    A = numpy.array(A).astype(float)

    target_irr, cash_flows = A[0], A[1:]

    data = [c for c in cash_flows]
    data.extend([target_irr])
    x = fsolve(f_irr, 0, args=tuple(data))[0]
    data[-1] = x
    check = numpy_financial.irr(data)

    # print(cash_flows)
    # print('target IRR: ',target_irr)
    # print('sale: ',x)
    # print('check: ',check)

    return float(x)
# ----------------------------------------------------------------------------------------------------------------------
def custom_func_read_file(filename:str):
    log_file = './data/output/custom_func_read_file.txt'
    mode = 'a' if os.path.isfile(log_file) else 'w'
    with open(log_file, mode) as f:
        f.write('\n'+filename)

    res = ''
    filename=str(filename)
    split = filename.split('"')
    if len(split) > 1:
        filename2 = split[1]
    else:
        split = filename.split("'")
        if len(split) > 1:
            filename2 = split[1]
        else:
            filename2 = filename.replace('\n', '')
            filename2.replace('"','')
            filename2 = re.sub("[^0-9a-zA-Z.]", "", filename2)

    filename2 = filename2.split('.')[0]
    filename2 = './data/ex_datasets/log_files/'+filename2+'.txt'
    with open(log_file, 'a') as f:
        f.write('\n'+filename2)

    if os.path.exists(filename2):
        with open(log_file,  'a') as f:f.write('\nOK')
        with open(filename2, 'r') as f:res = str(f.read())
    return res
# ----------------------------------------------------------------------------------------------------------------------
def get_tool_calc(LLM):
    calculator = LLMMathChain.from_llm(llm=LLM)
    #tools = [Tool(nfunc=calculator.run,name="Calculator",description=f"""Useful when you need to do math operations or arithmetic.""")]
    #tools = [StructuredTool.from_function(func=calculator.run,name="Calculator",description="Useful when you need to do math operations or arithmetic")]
    tools = [Tool(func=calculator.run, name="Calculator",description="Useful when you need to do math operations or arithmetic")]
    return tools
# ----------------------------------------------------------------------------------------------------------------------
def get_tool_klarna(LLM):
    tools = NLAToolkit.from_llm_and_url(LLM,"https://www.klarna.com/us/shopping/public/openai/v0/api-docs/").get_tools()
    return tools
# ----------------------------------------------------------------------------------------------------------------------
def get_tool_IRR():
    tools = [StructuredTool.from_function(func=custom_func_IRR_calc,name="IRR calculator", description="Calculate IRR from cash flow string")]
    return tools
# ----------------------------------------------------------------------------------------------------------------------
def get_tool_sale_for_target_IRR():
    tools =  [StructuredTool.from_function(func=custom_func_sales_for_target_irr_single, name="Sales for IRR target calculator",description="Calculate a sale to achieve a specific IRR from given cash flow string")]
    #tools = [StructuredTool.from_function(func=custom_func_sales_for_target_irr, name="Sales for IRR target calculator",description="Calculate a sale to achieve a specific float IRR from given cash flow array of floats")]
    return tools
# ----------------------------------------------------------------------------------------------------------------------
def get_tools_pandas_v01(df):
    python = PythonAstREPLTool(locals={"df": df})
    tools = [Tool(func=python.run,name="Pandas Tool", description=f"Tool to answer questions about pandas dataframe 'df'. Run python pandas operations on 'df' that has the following columns: {df.columns.to_list()}")]
    return tools
# ----------------------------------------------------------------------------------------------------------------------
def get_tools_pandas_v02(df):
    python = PythonAstREPLTool(locals={"df": df})
    tools = [StructuredTool.from_function(func=python.run,name="Pandas Tool", description=f"Tool to answer questions about pandas dataframe 'df'. using tool_input key, run python pandas operations on 'df' that has the following columns: {df.columns.to_list()}")]
    return tools
# ----------------------------------------------------------------------------------------------------------------------
def get_tool_age_of_Alice():
    def custom_func_Alice_age(year: str):return int(int(year)-1979)
    tools = [StructuredTool.from_function(func=custom_func_Alice_age, name="age of Alice",description="Calculate an age of Alice given provided year")]
    return tools
# ----------------------------------------------------------------------------------------------------------------------
def get_tool_age_of_Bob():
    def custom_func_Bob_age(year: str): return int(int(year)-2008)
    tools = [StructuredTool.from_function(func=custom_func_Bob_age, name="age of Bob",description="Calculate an age of Bob given provided year")]
    return tools
# ----------------------------------------------------------------------------------------------------------------------
def get_tool_read_file():
    tools =  [StructuredTool.from_function(func=custom_func_read_file, name="Tool to read the file",description="Returns file content based on filename")]
    return tools
# ----------------------------------------------------------------------------------------------------------------------
# def get_tool_analyze_issue():
#     tools = [Tool(func=A_RAG.run_query, name="File Analyzer", description="Automated analysis of the content retreived by the file reader tool.")]
#     return tools
# ----------------------------------------------------------------------------------------------------------------------
#def get_tool_search():
#    tools = [TavilyAnswer(max_results=1, name="Intermediate Answer")]
#    return tools
# ----------------------------------------------------------------------------------------------------------------------
def pretify_output(plain_text):

    lines = plain_text.split('\n')
    for line in lines:
        items = line.split('')

    return
# ----------------------------------------------------------------------------------------------------------------------
