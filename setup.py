from distutils.core import setup

if __name__== '__main__':
    setup(include_package_data=True,
          description='TF MOtif Discovery from Importance SCOres',
          url='NA',
          download_url='NA',
          version='0.4.1.0',
          packages=['modisco', 'modisco.cluster','modisco.backend',
                    'modisco.visualization', 'modisco.affinitymat',
                    'modisco.tfmodisco_workflow', 'modisco.hit_scoring'],
          setup_requires=[],
          install_requires=['numpy>=1.9', 'joblib>=0.11', 
                            'scikit-learn>=0.19',
                            'h5py>=2.5'],
          extras_require={
            'tensorflow': ['tensorflow>=1.7'],
            'tensorflow with gpu': ['tensorflow-gpu>=1.7']},
          scripts=[],
          name='modisco')
