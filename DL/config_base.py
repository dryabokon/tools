import os
import argparse
import json
import yaml
# ----------------------------------------------------------------------------------------------------------------------
class cnfg_base(object):
    def __init__(self, filename_in=None,do_parsing=True):
        if do_parsing:
            self.parser = argparse.ArgumentParser()
            self.init(filename_in)
        return

    # ----------------------------------------------------------------------------------------------------------------------
    def init(self, filename_in):
        if filename_in is not None:
            with open(filename_in, 'r') as file:
                config = yaml.safe_load(file)
                for key, value in config.items():
                    setattr(self, key, None if str(value) == 'None' else value)

        for key, default in zip(*self.get_keys_values()):
            if isinstance(default, bool):
                self.parser.add_argument(f'--{key}',type=self.smart_bool,default=default)
            elif default is None:
                self.parser.add_argument(f'--{key}', type=self.smart_cast, default=None)
            else:
                self.parser.add_argument(f'--{key}', type=type(default), default=default)

        args = self.parser.parse_args()
        for key in self.get_keys():
            if hasattr(args, key):
                setattr(self, key, getattr(args, key))
        return
    # ----------------------------------------------------------------------------------------------------------------------
    def smart_bool(self, s):
        if isinstance(s, bool):return s
        s = str(s).lower()
        if s in ("1", "true", "yes", "y"):return True
        elif s in ("0", "false", "no", "n"):return False
        else:raise argparse.ArgumentTypeError(f"Invalid boolean value: {s}")
    # ----------------------------------------------------------------------------------------------------------------------
    def smart_cast(self, s: str):
        if s.lower() in ("true", "false"): return s.lower() == "true"
        try:
            return int(s)
        except:
            pass
        try:
            return float(s)
        except:
            pass
        return s

    # ----------------------------------------------------------------------------------------------------------------------
    def save(self, filename_out):
        with open(filename_out, 'w') as f:
            json.dump(
                {k: getattr(self, k) for k in dir(self) if not (k.startswith("__") or callable(getattr(self, k)))}, f,
                indent=4)

    # ----------------------------------------------------------------------------------------------------------------------
    def get_keys(self):
        return [attr for attr in dir(self) if not (attr.startswith("__") or callable(getattr(self, attr)))]

    # ----------------------------------------------------------------------------------------------------------------------
    def get_keys_values(self):
        keys, values = [], []
        for k in dir(self):
            if not (k.startswith("__") or callable(getattr(self, k))):
                keys.append(k)
                values.append(getattr(self, k))
        return keys, values

    # ----------------------------------------------------------------------------------------------------------------------
    def print(self):
        for k, v in zip(*self.get_keys_values()):
            print(f'{k}: {v}')
        return

    # ----------------------------------------------------------------------------------------------------------------------
    def patch_path(self,path):
        path_patched = path
        if not os.path.isfile(path):
            if path[:2] == './':
                path_patched = '.' + path
                if not os.path.isfile(path_patched):
                    path_patched = '../' + path_patched


            elif path[:1] != '.':
                if os.path.isfile('..' + path):
                    path_patched = '..' + path
                elif os.path.isfile('../' + path_patched):
                    path_patched = '../' + path_patched

        return path_patched

    # ----------------------------------------------------------------------------------------------------------------------
    def save_as_python_script(self,filename_out):
        KV = [(k, v) for k, v in zip(*self.get_keys_values()) if k != 'parser']
        with open(filename_out, 'w') as f:
            f.write('from DL import config_base\n')
            f.write('class cnfg_experiment(config_base.cnfg_base):\n')
            for k, v in KV:
                if isinstance(v, (bool, type(None))):
                    v = str(v)
                elif isinstance(v, (int, float)):
                    v = str(v)
                else:
                    v = f"'{v}'"
                f.write(f'\t{k} = {v}\n')

        return
    # ----------------------------------------------------------------------------------------------------------------------