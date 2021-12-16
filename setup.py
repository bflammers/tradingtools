
from setuptools import setup

setup(name='tradingtools',
      version='0.0.1',
      description='Tools for trading stuff',
      author='Bart Lammers',
      author_email='bart.f.lammers@gmail.com',
      install_requires=[
          'cryptofeed[redis]~=2.1.1',
          'ccxt~=1.64.8',
          'numpy~=1.21.4',
          'polars~=0.11.0'
      ],
      packages=['tradingtools'],
      zip_safe=False)
