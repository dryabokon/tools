import os.path
import pandas as pd
# ---------------------------------------------------------------------------------------------------------------------
import tools_DF
# ---------------------------------------------------------------------------------------------------------------------
def export(filename_in,folder_out,skiprows=0,sep=',',pretify=False):
    if not os.path.isdir(folder_out):
        os.mkdir(folder_out)

    xls = pd.ExcelFile(filename_in)
    sheet_names = xls.sheet_names
    multi_sheet = True if len(sheet_names)>1 else False

    for sheet_name in sheet_names:
        split = filename_in.split('.')
        filename_out,ext_out = split[-2],split[-1]
        filename_out = filename_out.split('/')[-1]

        if multi_sheet:
            #filename_out+='_'+sheet_name
            filename_out=sheet_name

        df = pd.read_excel(filename_in, engine='openpyxl', sheet_name=sheet_name,skiprows=skiprows)
        if pretify:
            with open(folder_out + filename_out+'.txt', 'w', encoding='utf-8') as f:
                txt = tools_DF.prettify(df, showindex=False, tablefmt='psql')
                f.write(txt)
        else:
            df.to_csv(folder_out + filename_out+'.csv',sep=sep,index=False)
    return
# ---------------------------------------------------------------------------------------------------------------------
def load(filename_in,skiprows=None,nrows=None):

    dfs = []
    xls = pd.ExcelFile(filename_in)
    for sheet_name in xls.sheet_names:
        df = pd.read_excel(filename_in, engine='openpyxl', skiprows=skiprows,sheet_name=sheet_name, nrows=nrows)
        dfs.append(df)

    if len(dfs)==1:
        dfs = dfs[0]
    return dfs
# ---------------------------------------------------------------------------------------------------------------------
def save(filename_out,dataframes,sheet_names):

    writer = pd.ExcelWriter(filename_out, engine='openpyxl')
    for df,sheet_name in zip(dataframes,sheet_names):
        df.to_excel(writer, sheet_name=sheet_name,index=False)

    writer.save()
    return
# ---------------------------------------------------------------------------------------------------------------------