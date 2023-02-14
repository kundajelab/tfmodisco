from setuptools import setup

setup(
	name='modisco-lite',
	version='2.0.6',
	author='Jacob Schreiber',
	author_email='jmschreiber91@gmail.com',
	packages=['modiscolite'],
	scripts=['modisco'],
	url='https://github.com/jmschrei/tfmodisco-lite',
	license='LICENSE.txt',
	description='Transcription Factor MOtif Discovery from Importance SCOres - lite',
	install_requires=[
		'numpy >= 1.21.5', 
		'scipy >= 1.6.2',
		'numba >= 0.53.1',
		'scikit-learn >= 1.0.2',
		'leidenalg == 0.8.10',
		'igraph == 0.9.11',
		'tqdm >= 4.38.0',
		'pandas >= 1.4.3',
		'logomaker >= 0.8',
		'h5py >= 3.7.0',
		'hdf5plugin'
	]
)
