from setuptools import find_packages
from distutils.core import setup

if __name__== '__main__':
    setup(include_package_data=True,
          description='MOtif Discovery from Importance SCOres - lite',
          long_description="""Algorithm for discovering consolidated patterns from base-pair-level importance scores""",
          url='https://github.com/jmschrei/tfmodisco-lite',
          version='0.0.1',
          packages=find_packages(),
          package_data={
                '': ['cluster/phenograph/louvain/*convert*', 'cluster/phenograph/louvain/*community*', 'cluster/phenograph/louvain/*hierarchy*']
          },
          zip_safe=False,
          setup_requires=[],
          install_requires=['numpy>=1.9', 'joblib>=0.11', 
                            'scikit-learn>=0.19',
                            'h5py>=2.5', 'leidenalg>=0.8.7',
                            'tqdm>=4.38.0', 'psutil>=5.4.8',
                            'matplotlib>=2.2.5'],
          extras_require={
            'tensorflow': ['tensorflow>=1.7'],
            'tensorflow with gpu': ['tensorflow-gpu>=1.7'],
            'pynn': ['pynndescent']},
          scripts=['modiscolite/cluster/run_leiden'],
          name='modiscolite')
