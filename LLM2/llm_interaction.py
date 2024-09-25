from PIL import Image as PImage
import pandas as pd
from halo import Halo
from langchain.text_splitter import RecursiveCharacterTextSplitter,CharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
# ----------------------------------------------------------------------------------------------------------------------
import tools_console_color
import tools_DF
import tools_time_profiler
# ----------------------------------------------------------------------------------------------------------------------
TP = tools_time_profiler.Time_Profiler()
# ----------------------------------------------------------------------------------------------------------------------
def from_list_of_dict(list_of_dict, select, as_df):
    if as_df:
        df = pd.DataFrame(list_of_dict)
        df = df.iloc[:, [c.find('@') < 0 for c in df.columns]]
        if select is not None and df.shape[0] > 0:
            df = df[select]
        result = df
    else:
        if isinstance(select, list):
            result = [';'.join([x + ':' + str(r[x]) for x in select]) for r in list_of_dict]
        else:
            if select is not None:
                result = [r[select] for r in list_of_dict]
            else:
                result = '\n'.join([str(r) for r in list_of_dict])
    return result
# ----------------------------------------------------------------------------------------------------------------------
def file_to_texts(filename_in,chunk_size=2000):
    with open(filename_in, 'r',encoding='utf8', errors ='ignore') as f:
        text_document = f.read()
    texts = RecursiveCharacterTextSplitter(separators=["\n\n", "\n"], chunk_size=chunk_size, chunk_overlap=100).create_documents([text_document])
    texts = [text.page_content for text in texts]
    return texts
# ----------------------------------------------------------------------------------------------------------------------
def pdf_to_texts(filename_pdf,chunk_size=2000):

    loader = PyPDFLoader(filename_pdf)
    pages = loader.load_and_split()
    text_splitter = CharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=100)
    docs = text_splitter.split_documents(pages)
    texts = [t for t in set([doc.page_content for doc in docs])]
    return texts
# ----------------------------------------------------------------------------------------------------------------------
def pdf_to_texts_and_images(filename_pdf,chunk_size=2000):
    from pdf2image import convert_from_path
    images = convert_from_path(filename_pdf)
    loader = PyPDFLoader(filename_pdf)
    pages = loader.load_and_split()
    texts = [page.page_content for page in pages]

    images = [img.resize((640, int(640*img.size[1]/img.size[0]))) for img in images]
    images = [PImage.merge("RGB", (img.split())) for img in images]

    return texts,images
# ----------------------------------------------------------------------------------------------------------------------

def pretify_string(text,N=80):
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
def display_res(res,width):
    if isinstance(res, pd.DataFrame):
        print(tools_DF.prettify(res, showindex=False))
    else:
        print(pretify_string(res, N=width))
    return
# ----------------------------------------------------------------------------------------------------------------------
def display_debug_info(texts):
    if len(texts)>0:
        print(tools_console_color.apply_style(pretify_string(texts[0]), color='blk'))
        for t in texts[1:]:
            print(tools_console_color.apply_style(''.join(['-'] * 20), color='blk'))
            print(tools_console_color.apply_style(pretify_string(t), color='blk'))
    return
# ----------------------------------------------------------------------------------------------------------------------
def interaction_offline(A,query,do_debug=False,do_spinner=False):
    width = 80
    if not isinstance(query,list):
        query = [query]

    for q in query:
        print(tools_console_color.apply_style(q,style='BLD'))
        if do_spinner:
            TP.tic('xxx', verbose=False,reset=True)
            spinner = Halo(spinner={'interval': 100, 'frames': ['⠋', '⠙', '⠹', '⠸', '⠼', '⠴', '⠦', '⠧', '⠇', '⠏']})
            spinner.start()

        r = A.run_query(q)
        if len(r) ==2:
            res, texts = r[0],r[1]
        else:
            res,texts = r,[]

        if do_spinner:
            spinner.stop()
            spinner.succeed(TP.print_duration('xxx', verbose=False))

        display_res(res,width)
        if do_debug:display_debug_info(texts)
        if len(query)>1:
            print(''.join(['=']*width))

    return r
# ----------------------------------------------------------------------------------------------------------------------
def interaction_live(A,method='run_query',do_debug=False,do_spinner=False):
    width = 80
    should_be_closed = False
    while not should_be_closed:
        print(''.join(['='] * width))
        print(tools_console_color.apply_style('>','GRN'),end='')
        query = input()
        if len(query)==0:
            should_be_closed = True
            continue

        if do_spinner:
            TP.tic('xxx', verbose=False,reset=True)
            spinner = Halo(spinner={'interval': 100, 'frames': ['⠋', '⠙', '⠹', '⠸', '⠼', '⠴', '⠦', '⠧', '⠇', '⠏']})
            spinner.start()

        #res,texts = A.run_query(query)
        r = getattr(A, method)(query)
        if len(r) ==2:
            res, texts = r[0],r[1]
        else:
            res,texts = r,[]

        if do_spinner:
            spinner.stop()
            spinner.succeed(TP.print_duration('xxx', verbose=False))


        display_res(res, width)
        if do_debug:display_debug_info(texts)

    return
# ----------------------------------------------------------------------------------------------------------------------