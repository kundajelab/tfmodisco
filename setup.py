from distutils.core import setup
from setuptools import find_packages

if __name__== '__main__':
    setup(include_package_data=True,
          description='TF MOtif Discovery from Importance SCOres',
          long_description="""Algorithm for discovering consolidated patterns from base-pair-level importance scores""",
          url='https://github.com/kundajelab/tfmodisco',
          version='0.5.5.3',
          packages=find_packages(),
          package_data={
                '': ['cluster/phenograph/louvain/*convert*', 'cluster/phenograph/louvain/*community*', 'cluster/phenograph/louvain/*hierarchy*']
          },
          zip_safe=False,
          setup_requires=[],
          install_requires=['numpy>=1.9', 'joblib>=0.11', 
                            'scikit-learn>=0.19', 'numba>=0.43.1',
                            'h5py>=2.5', 'tqdm>=4.41.0'],
          scripts=[],
          name='modisco')
