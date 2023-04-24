import os
from setuptools import setup, find_packages


CLASSIFIERS = [
    'Operating System :: Ubuntu',
    'Intended Audience :: Science/Research',
    'Programming Language :: Python',
    'Programming Language :: Python :: 2',
    'Programming Language :: Python :: 2.7',
    'Programming Language :: Python :: 3',
    'Programming Language :: Python :: 3.4',
    'Programming Language :: Python :: 3.5',
    'Topic :: Scientific/Engineering',
]


setup(name='score_ensemble',
      description='Data processing code for',
      long_description=(open('README.rst').read()
                        if os.path.exists('README.rst')
                        else ''),
      version='0.1',
      license='Apache',
      classifiers=CLASSIFIERS,
      author='Clement Brochet',
      author_email='clement.brochet@meteo.fr',
      install_requires=['numpy', 'pandas', 'metrics4ensemble'],
      tests_require=[],
      packages=find_packages())
