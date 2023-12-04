import inspect
import os.path
import numpy
from git import Repo
import pandas as pd
# ---------------------------------------------------------------------------------------------------------------------
import tools_console_color

from LLM2 import llm_models
from LLM2 import llm_chains
# ---------------------------------------------------------------------------------------------------------------------
class Analizer_git(object):
    def __init__(self,llm_cnfg,repo_url,folder_out):
        self.folder_out = folder_out
        self.repo_url = repo_url
        self.repo = self.init_repo()

        LLM = llm_models.get_model(llm_cnfg.filename_config_chat_model, model_type='QA')
        self.chain = llm_chains.get_chain_chat(LLM)

        return
# ---------------------------------------------------------------------------------------------------------------------
    def init_repo(self):
        if os.path.isdir(self.folder_out + self.repo_url.split('/')[-1]):
            repo = Repo(self.folder_out + self.repo_url.split('/')[-1])
        else:
            repo = Repo.clone_from(self.repo_url, self.folder_out + self.repo_url.split('/')[-1])
        return repo
# ---------------------------------------------------------------------------------------------------------------------
    def function_signature_string(self,member):
        parameters = inspect.signature(member).parameters
        return member.__name__ + "(" + ', '.join(str(x) for x in parameters.values()) + ")"
# ---------------------------------------------------------------------------------------------------------------------
    def recusive_module_search(self,module):
        members = inspect.getmembers(module)

        for name, member in members:
            if inspect.ismodule(member):
                # Dont go too deep :)
                if member is module:
                    self.recusive_module_search(member)
            elif inspect.isfunction(member):
                file = inspect.getfile(member)
                print(file, self.function_signature_string(member), "function")
            elif inspect.isclass(member):
                file = inspect.getfile(member)
                print(file, self.function_signature_string(member), "class")
                class_members = inspect.getmembers(member)
                for name, class_member in class_members:
                    if inspect.isfunction(class_member):
                        member_args = inspect.signature(class_member)
                        print(file, member.__name__ + "." + self.function_signature_string(class_member), "method")

        return
# ---------------------------------------------------------------------------------------------------------------------
    def get_repo_structure_tree(self,root=None, level=0, res=None, as_txt=True):
        chr_trace = " "
        chr_bar =  '｜'

        if root is None:
            root = self.repo.tree()

        idx = numpy.argsort([entry.type for entry in root])[::-1]
        entries = [e for e in root]
        entries = [entries[i] for i in idx]

        if res is None:
            if as_txt:
                res = ''
            else:
                res =pd.DataFrame([])

        for entry in entries:
            if entry.type=='tree':
                prefix, suffix = tools_console_color.uni_folder + ' ' + tools_console_color.dct_style['UND']+tools_console_color.dct_style['BLD'],tools_console_color.ENDC
            else:
                prefix, suffix = '',''

            if as_txt:
                res+= f'{chr_trace * 4 * level}' + chr_bar + f'{prefix}{entry.path}{suffix}' + '\n'
                if entry.type == "tree":
                    res= self.get_repo_structure_tree(entry,level + 1,res,as_txt)
            else:
                res = pd.concat([res,pd.DataFrame({'level':[level],'type':[entry.type],'path':[entry.path]})])
                if entry.type == "tree":
                    res = self.get_repo_structure_tree(entry,level + 1,res,as_txt)

        return res
    # ---------------------------------------------------------------------------------------------------------------------
    def get_history(self,max_count=None):
        # git show 12ecc47
        #git show --stat 12ecc47

        prev_commits = list(self.repo.iter_commits(all=True, max_count=max_count))
        res = [[c.hexsha[:7], c.committed_datetime, c.committer.name, c.message] for c in prev_commits]
        df = pd.DataFrame(res,columns=['hash','date','author','message'])
        df['d_files'] = 0
        df['d_lines'] = 0

        for i in range(len(prev_commits)):
            df_simple   = self.get_diff(base=i,back=i+1,create_patch=False)
            df_detailed = self.get_diff(base=i,back=i+1, create_patch=True)
            df.iloc[i,-2] = df_simple.shape[0]
            df.iloc[i,-1] = df_detailed.iloc[:,-1].str.count('\n').sum()

        df = df[['hash','date','author','d_files','d_lines','message']]
        df['date'] = df['date'].apply(lambda x: x.strftime('%Y-%m-%d:%H:%M:%S'))

        return df
    # ---------------------------------------------------------------------------------------------------------------------
    def get_diff(self,base=0,back=1,create_patch=False):
        # git diff 29b26a1
        # git diff --name-only 29b26a1
        prev_commits = list(self.repo.iter_commits(all=True))

        df = pd.DataFrame([])
        if back<len(prev_commits):
            diffs = prev_commits[base].diff(prev_commits[back],create_patch=create_patch)
            if create_patch:
                df = pd.DataFrame([[diff.a_path,str(diff.diff).replace("\\n","\n")] for diff in diffs], columns=['path','diff'])
                df.rename(columns={'diff': prev_commits[base].hexsha[:7]}, inplace=True)
            else:
                df = pd.DataFrame([[diff.change_type, diff.a_path] for diff in diffs],columns=['type', 'path'])
                map = {'A': 'Added', 'C': 'Copied', 'D': 'Deleted', 'M': 'Modified', 'R': 'Renamed','T': 'have their type (mode) changed', 'U': 'Unmerged', 'X': 'Unknown','B': 'have had their pairing Broken'}
                df['type'] = df['type'].map(map)

        return df
    # ---------------------------------------------------------------------------------------------------------------------
    def summarize_commit(self,base=0,back=1,detailed=True):
        OPEN_AI_PROMPT = 'The following is a git diff of a single file. Please summarize it in a comment, ' \
                         'describing the changes made in the diff in high level. Do it in the following way: Write \`SUMMARY:\` and then write a ' \
                         'summary of the changes made in the diff, as a bullet point list. Every bullet point should start with a \`*\`.'
        df = self.get_diff(base, back,create_patch=True)
        df['summary'] = ''
        for r in range(df.shape[0]):
            filename = df['path'].iloc[r]
            patch = df.iloc[r,-2]
            Q = f'{OPEN_AI_PROMPT}\n\nTHE GIT DIFF OF ${filename} TO BE SUMMARIZED:\n\`\`\`\n${patch}\n\`\`\`\n\nSUMMARY:\n'
            summary = self.chain.run(question=Q, input_documents=[])
            df['summary'].iloc[r] = summary

        #df = df.iloc[:,1:]

        OPEN_AI_PROMPT = 'You are an expert programmer, and you are trying to summarize a pull request.\
        You went over every commit that is part of the pull request and over every file that was changed in it.\
        For some of these, there was an error in the commit summary, or in the files diff summary.\
        Please summarize the pull request. Write your response in bullet points, starting each bullet point with a \`*\`.\
        Write a high level description. Do not repeat the commit summaries or the file summaries.\
        Write the most important bullet points. The list should not be more than a few bullet points.'

        filesString = '\n'.join(df['summary'].values.tolist())
        Q = f'{OPEN_AI_PROMPT}\n\nTHE FILE SUMMARIES:\n\`\`\`\n${filesString}\n\`\`\`\n\nReminder - write only the most important points. No more than a few bullet points.THE PULL REQUEST SUMMARY:\n'
        summary = self.chain.run(question=Q, input_documents=[])

        df_summary = pd.DataFrame({'path':'Summary','summary':[summary]})
        if detailed:
            df = pd.concat([df,df_summary])
        else:
            df = df_summary
        return df
    # ---------------------------------------------------------------------------------------------------------------------
    def summarize_commits(self,max_count=None):
        df = self.get_history(max_count=max_count)
        df = df[['hash', 'd_files', 'd_lines', 'message']]
        df['AI_summary'] = ''
        for i in range(df.shape[0]-1):
            df_s = self.summarize_commit(base=i, back=i+1, detailed=False)
            summary = df_s['summary'].iloc[0]
            #summary = '\n\n* '.join([fill(x, width=maxcolwidths) for x in summary.split('* ')])[1:]
            df.iloc[i, -1] = summary

        return df
# ---------------------------------------------------------------------------------------------------------------------


