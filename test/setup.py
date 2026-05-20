from setuptools import setup, find_packages

setup(
    name="ezclean",
    version="0.1.0",
    description="An industrial-standard automated data cleaning, inspection, and visualization library.",
    author="Thilac01",
    author_email="example@example.com",
    packages=find_packages(),
    install_requires=[
        "pandas>=1.3.0",
        "numpy>=1.20.0",
        "scikit-learn>=1.0.0",
        "scipy>=1.7.0",
        "plotly>=5.0.0",
        "requests",
        "pypdf"
    ],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Topic :: Scientific/Engineering :: Information Analysis",
    ],
    python_requires=">=3.7",
)
