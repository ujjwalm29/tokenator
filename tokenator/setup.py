from setuptools import setup, find_packages

setup(
    name="tokenator",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "openai>=1.3.0",
        "sqlalchemy>=2.0.0",
    ],
    author="Your Name",
    author_email="your.email@example.com",
    description="Token usage tracking and cost analysis for OpenAI APIs",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/tokenator",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.7",
) 