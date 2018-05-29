'''Cloud ML Engine package configuration.'''
from setuptools import setup, find_packages

setup(name='semantic_segmenter',
      version='1.0',
      packages=find_packages(),
      include_package_data=True,
      description='Semantic segmentation',
      author='Rohith',
      author_email='rohith.menon@zee.aero',
      license='Unlicense',
      install_requires=[
          'keras==2.1.3',
          'h5py'],
      zip_safe=False)
