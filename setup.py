
from setuptools import setup

setup(name='tradingtools',
      version='0.0.1',
      description='Tools for trading stuff',
      author='Bart Lammers',
      author_email='bart.f.lammers@gmail.com',
      install_requires=[
          'pandas',
          'numpy'
      ],
      packages=['tradingtools'],
      zip_safe=False)
