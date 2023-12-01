# ----------------------------------------------------------------------------------------------------------------------
dct_style = { 'BLD':'\33[1m',
              'DIM':'\33[2m',
              'CUR':'\33[3m',
              'UND':'\33[4m',
              'BLN':'\33[5m',
              'BLN2':'\33[6m',
              'INV':'\33[7m'}
dct_col_fg ={ 'BLK':'\033[30m', 'blk':'\033[90m',
              'RED':'\033[31m', 'red':'\033[91m',
              'GRN':'\033[32m', 'grn':'\033[92m',
              'YLW':'\033[33m', 'ylw':'\033[93m',
              'BLU':'\033[34m', 'blu':'\033[94m',
              'MGN':'\033[35m', 'mgn':'\033[95m',
              'CYN':'\033[36m', 'cyn':'\033[96m',
              'WHT':'\033[37m', 'wht':'\033[97m'}
dct_col_bg ={'BLK':'\x1b[40m', 'blk':'\x1b[100m',
             'RED':'\x1b[41m', 'red':'\x1b[101m',
             'GRN':'\x1b[42m', 'grn':'\x1b[102m',
             'YLW':'\x1b[43m', 'ylw':'\x1b[103m',
             'BLU':'\x1b[44m', 'blu':'\x1b[104m',
             'MGN':'\x1b[45m', 'mgn':'\x1b[105m',
             'CYN':'\x1b[46m', 'cyn':'\x1b[106m',
             'WHT':'\x1b[47m', 'wht':'\x1b[107m'}

ENDC    = '\033[0m'
ENDB    = '\x1b[0m'
# ----------------------------------------------------------------------------------------------------------------------
uni_folder = u'\U0001F5C0'
# ----------------------------------------------------------------------------------------------------------------------
def apply_style(str,color=None,background=None,style=None):
    if color is not None and color in dct_col_fg.keys():
        str = dct_col_fg[color] + str + ENDC

    if background is not None and background in dct_col_bg.keys():
        str = dct_col_bg[background] + str + ENDB

    if style is not None and style in dct_style.keys():
        str = dct_style[style] + str + ENDC

    return str
# ----------------------------------------------------------------------------------------------------------------------
def get_test_string():

    res  = [apply_style(k, color=k) for k in dct_col_fg.keys()]
    res += ['   ']
    res += [apply_style(k, background=k) for k in dct_col_bg.keys()]
    res += ['   ']
    res +=[apply_style(k, style=k) for k in dct_style.keys()]

    res = '\n'.join(res)
    return res
# ----------------------------------------------------------------------------------------------------------------------
def highlight_words(text,word,color='blue',background=None,is_bold=True,is_underline=False):
    if not isinstance(word,list):
        word = [word]
    for w in word:
        style = 'BLD' if is_bold else ''
        style += 'UND' if is_underline else ''
        text = text.replace(w,apply_style(w,color=color,background=background,style=style))
    return text
# ----------------------------------------------------------------------------------------------------------------------