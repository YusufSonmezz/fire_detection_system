from setuptools import setup, find_packages
from typing import List

HYPHEN_E_DOT = '-e .'

def get_requirements(file_path:str)->List[str]:
    requirements = []
    with open(file_path) as file_obj:
        requirements = file_obj.readlines()
        requirements = [req.replace("\n", "") for req in requirements]

        if HYPHEN_E_DOT in requirements:
            requirements.remove(HYPHEN_E_DOT)
    
    return requirements

setup(
    name="fire_detector_via_drone",
    version="0.0.1",
    author="Yusuf Sonmez",
    author_email="yusuf.sonmez1@ogr.sakarya.edu.tr",
    packages=find_packages(),
    install_requires=get_requirements('requirements.txt')
)