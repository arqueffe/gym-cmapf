from setuptools import setup, find_packages

setup(
    name='gym_cmapf',
    version='0.0.1',
    packages=find_packages(),

    author='Arthur Queffelec',
    author_email='arthur.queffelec@gmail.com',

    install_requires=[
        'gym>=0.17.3'
    ]
)