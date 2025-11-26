# scripts/download_all_data.py
import sys
import os

# Add project root to sys.path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, project_root)

from src.data.download_finance import download_finance_all

def main():
    download_finance_all()


if __name__ == "__main__":
    main()

