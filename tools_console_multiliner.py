import sys
import os

class MultiLineUpdater:
    def __init__(self, num_rows):
        if num_rows is None or num_rows==0:
            return
        self.num_rows = num_rows
        self.lines = [""] * num_rows
        if os.name == 'nt':
            os.system('')
        print("\n" * (num_rows - 1), end="")
        sys.stdout.flush()
        return
    # ----------------------------------------------------------------------------------------------------------------------
    def update(self, row: int, text: str):
        self.lines[row] = text
        rows_up = self.num_rows - 1 - row
        sys.stdout.write(f"\033[{rows_up}A\r" if rows_up > 0 else "\r")
        sys.stdout.write(f"\033[2K{text}")
        if rows_up > 0:
            sys.stdout.write(f"\033[{rows_up}B")
        sys.stdout.flush()
    # ----------------------------------------------------------------------------------------------------------------------
    def close(self):
        sys.stdout.write("\n")
        sys.stdout.flush()
# ----------------------------------------------------------------------------------------------------------------------