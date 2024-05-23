from setuptools import setup, find_packages

setup(
    name='envs_robot',
    version='0.0.1',
    packages=find_packages(include=['envs_robot', 'envs_robot.*'])
)