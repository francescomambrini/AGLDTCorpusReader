#!/usr/bin/env python

from setuptools import setup

setup(name='agldt_corpus_reader',
      version='0.1',
      description='A Python-NLTK-like corpus reader for XML files that comply to the formats used by Perseus\' Ancient Greek and Latin Dependency Treebank',
      url='',
      author='Francesco Mambrini',
      author_email='',
      license='CC BY SA',
      packages=['agldt_corpus_reader'],
      install_requires=[
          'nltk',
          'lxml'
      ],
      include_package_data=True,
      zip_safe=False)
