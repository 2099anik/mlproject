from setuptools import setup, find_packages
from typing import List


HYPEN_E_DOT='-e .'

def get_requirements(file_path:str)->list[str]:
    requirements=[]
    with open(file_path,'r') as f:
        requirements=f.readlines()
        requirements=[req.replace('\n','') for req in requirements]
        
        if HYPEN_E_DOT in requirements:
            requirements.remove(HYPEN_E_DOT)
            
    return requirements

setup(
    name='mlproject',
    version='0.0.1',
    author='Anik',
    author_email='anikdebb.ee21@rvce.edu.in',
    packages=find_packages(),
    install_requires=get_requirements('requirements.txt'),  # add any additional packages that
)