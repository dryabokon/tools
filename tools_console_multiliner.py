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
        self.lines[row] = text if text is not None else ''
    # ----------------------------------------------------------------------------------------------------------------------
    def redraw(self):
        sys.stdout.write(f"\033[{self.num_rows - 1}A\r")
        for i, line in enumerate(self.lines):
            sys.stdout.write(f"\033[2K{line}")
            if i < self.num_rows - 1:
                sys.stdout.write("\n")
        sys.stdout.flush()
    # ----------------------------------------------------------------------------------------------------------------------
    def reset(self):
        """Re-anchor the block at the current cursor position after stray prints."""
        print("\n" * (self.num_rows - 1), end="")
        sys.stdout.flush()
    # ----------------------------------------------------------------------------------------------------------------------
    def close(self):
        sys.stdout.write("\n")
        sys.stdout.flush()
# ----------------------------------------------------------------------------------------------------------------------