# setup.py
from setuptools import setup, find_packages

setup(
    name="human_ai_cognition",
    version="0.1.0",
    description="Human‑AI Cognition: where memory meets meaning",
    author="Kevin Lee Swaim",
    url="https://github.com/doodmeister/human-ai-cognition",
    packages=find_packages(exclude=["tests*", "docs*"]),
    python_requires=">=3.8",
    install_requires=[
        "torch>=1.13.0",
        "boto3",
        "streamlit",
        "plotly",
        # …add any other runtime dependencies here
    ],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
)
