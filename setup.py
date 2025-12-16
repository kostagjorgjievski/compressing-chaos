from setuptools import setup, find_packages

setup(
    name="compressing-chaos",
    version="0.1.0",
    description="Latent diffusion models for chaotic time series",
    packages=find_packages(),
    python_requires=">=3.8",
    install_requires=[
        "numpy",
        "pandas",
        "scipy",
        "torch>=2.0",
        "pytorch-lightning",
        "statsmodels",
        "pmdarima",
        "yfinance",
        "fredapi",
        "matplotlib",
        "seaborn",
        "scikit-learn",
        "tqdm",
        "tensorboard",
    ],
)
