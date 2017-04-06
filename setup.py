# encoding: utf-8
from setuptools import setup

setup(name='imaginet',
      version='1.0',
      description='Visually grounded word and sentence representations',
      url='https://github.com/gchrupala/visually-grounded-speech',
      author='Grzegorz Chrupała',
      author_email='g.chrupala@uvt.nl',
      license='MIT',
      packages=['imaginet'],
      install_requires=[
          'Theano',
          'funktional==0.6'
                    ],
      zip_safe=False)
