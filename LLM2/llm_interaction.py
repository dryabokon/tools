import pandas as pd
from halo import Halo
# ----------------------------------------------------------------------------------------------------------------------
import tools_console_color
import tools_DF
import tools_time_profiler
# ----------------------------------------------------------------------------------------------------------------------
TP = tools_time_profiler.Time_Profiler()
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