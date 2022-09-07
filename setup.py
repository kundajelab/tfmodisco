from setuptools import setup

setup(
	name='modiscolite',
	version='0.0.1',
	author='Jacob Schreiber',
	author_email='jmschreiber91@gmail.com',
	packages=['modiscolite'],
	url='https://github.com/jmschrei/tfmodisco-lite',
	license='LICENSE.txt',
	description='Transcription Factor MOtif Discovery from Importance SCOres - lite',
	long_description="""Algorithm for discovering consolidated patterns from base-pair-level importance scores""",
	install_requires=[
		'numpy >= 1.22.2', 
		'scipy >= 1.6.2',
		'numba >= 0.53.1',
		'scikit-learn >= 1.0.2',
		'leidenalg >= 0.8.7',
		'igraph >= 0.9.9',
		'tqdm >= 4.38.0',
	]
)
