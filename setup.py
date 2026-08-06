from setuptools import setup, find_packages

setup(
    name="hfsp",
    version="0.1.0",
    description="Hybrid Flow Shop Scheduling Problem Research Framework",
    author="Pengbo Gao",
    packages=find_packages(),
    install_requires=[
        "numpy>=1.24.0",
        "pandas>=2.0.0",
        "matplotlib>=3.7.0",
        "openpyxl>=3.1.0",
        "pyyaml>=6.0",
        "scipy>=1.10.0",
        "tqdm>=4.65.0",
    ],
    python_requires=">=3.9",
)
