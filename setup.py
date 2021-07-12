from setuptools import find_packages
from distutils.core import setup

if __name__== '__main__':
    setup(include_package_data=True,
          description='TF MOtif Discovery from Importance SCOres',
          long_description="""Algorithm for discovering consolidated patterns from base-pair-level importance scores""",
          url='https://github.com/kundajelab/tfmodisco',
          version='0.5.14.2',
          packages=find_packages(),
          package_data={
                '': ['cluster/phenograph/louvain/*convert*', 'cluster/phenograph/louvain/*community*', 'cluster/phenograph/louvain/*hierarchy*']
          },
          zip_safe=False,
          setup_requires=[],
          install_requires=['numpy>=1.9', 'joblib>=0.11', 
                            'scikit-learn>=0.19',
                            'h5py>=2.5', 'leidenalg>=0.7.0',
                            'tqdm>=4.38.0', 'psutil>=5.4.8',
                            'matplotlib>=2.2.5'],
          extras_require={
            'tensorflow': ['tensorflow>=1.7'],
            'tensorflow with gpu': ['tensorflow-gpu>=1.7']},
          scripts=['modisco/cluster/run_leiden'],
          name='modisco')
