# scripts/make_windows.py

import os
import sys

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from src.data.preprocessing import make_sp500_windows


def main():
    make_sp500_windows(seq_len=50)  # you can tweak seq_len later


if __name__ == "__main__":
    main()
