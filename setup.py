from setuptools import find_packages, setup

def readme():
    with open("ReadME.md") as f:
        return f.read()

setup(
    name="GluA2",
    version="0.0.1",
    description="Code for analyzing GluA2 data",
    long_description=readme(),
    long_description_content_type="text/markdown",
    url="https://github.com/wjakewright/GluA2",
    author="William (Jake) Wright, Jennifer Li",
    license="",
    packages=find_packages(),
)