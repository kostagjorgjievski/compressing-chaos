# scripts/make_windows_cond.py

import os
import sys

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from src.data.preprocessing_cond import make_sp500_windows_cond


def main():
    make_sp500_windows_cond(seq_len=50, name="sp500_logret")


if __name__ == "__main__":
    main()
