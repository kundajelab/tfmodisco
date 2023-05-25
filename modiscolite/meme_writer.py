# Create a class that can be used to write MEME files.
# Authors: Ivy Raine <ivy.ember.raine@gmail.com>, Jacob Schreiber <jmschreiber91@gmail.com>
from typing import List, Optional
import numpy as np

from os import PathLike

class MEMEWriterMotif:
	"""Class for handling Motif for MEME file writing.
	
	Parameters
	----------
	name : str
		The name of the motif.
	probability_matrix : np.ndarray
		The probability matrix of the motif.
	alphabet : str
		The alphabet of the motif. Used in calculating the alphabet length.
		However, it does not work with custom alphabets currently. 
	alphabet_length : Optional[int]
		The length of the alphabet. Also known as 'alength' in MEME Suite.
	source_sites : int
		The number of source sites. Also known as 'nsites' in MEME Suite.
	e_value : Optional[str]
		The E-value of the motif.
	url : Optional[str]
		The URL of the motif.
	"""

	def __init__(
		self,
		name: str,
		probability_matrix: np.ndarray,
		source_sites: int,
		alphabet: str,
		alphabet_length: Optional[int] = None,
		e_value: Optional[str] = None,
		url: Optional[str] = None
	) -> None:
		self._name = name
		self._probability_matrix = probability_matrix
		self._source_sites = source_sites
		self._alphabet_length = len(alphabet) if alphabet_length is None else alphabet_length
		self._e_value = e_value
		self._url = url

	@property
	def name(self):
		return self._name

	@property
	def probability_matrix(self):
		return self._probability_matrix

	@property
	def alphabet_length(self):
		return self._alphabet_length
	
	@property
	def source_sites(self):
		return self._source_sites

	@property
	def e_value(self):
		return self._e_value

	@property
	def url(self):
		return self._url

	def __repr__(self) -> str:
		return f"MEMEWriterMotif(name={self._name})"

	def __str__(self) -> str:
		output = (f'''\
MOTIF {self.name}
letter-probability matrix: alength= {str(self.alphabet_length)} w= {self.probability_matrix.shape[0]} nsites= {str(self.source_sites)}{f"E= {self.e_value}" if self.e_value is not None else ""}
{array_to_string(self.probability_matrix, 6)}''')
		if self.url is not None:
			output += f"URL {self.url}"	
		return output



class MEMEWriter:
	"""Class for writing MEME files based on MEME Suite specifications, from
	user-provided motifs.
	
	Parameters
	----------
	memesuite_version : str
		The version of MEME Suite used to create the file.
		May also be added incrementally using `add_motif()`.
	motifs : List[MEMEWriterMotif]
		The motifs to be written to the file.
	alphabet : Optional[str]
		The alphabet of the motifs. Optional but recommended.
		Example: "ACGT".
	background_frequencies : Optional[str]
		The background frequencies of the motifs.
		Example: "A 0.25 C 0.25 G 0.25 T 0.25".
	background_frequencies_source : Optional[str]
		The source of the background frequencies.
	strands : Optional[List[str]]
		The strands of the motifs. Optional.
		Example: "+ -".
	"""

	def __init__(
		self,
		memesuite_version: str,
		motifs: List[MEMEWriterMotif] = [],
		alphabet: Optional[str] = None,
		background_frequencies: Optional[str] = None,
		background_frequencies_source: Optional[str] = None,
		strands: Optional[List[str]] = None,
	) -> None:
		self._memesuite_version = memesuite_version
		self._motifs = motifs if motifs is not None else []
		self._alphabet = alphabet
		self._background_frequencies = background_frequencies
		self._background_frequencies_source = background_frequencies_source
		self._strands = strands

	@property
	def memesuite_version(self):
		return self._memesuite_version

	@property
	def motifs(self):
		return self._motifs
	
	@property
	def alphabet(self):
		return self._alphabet
	
	@property
	def background_frequencies(self):
		return self._background_frequencies

	@property
	def background_frequencies_source(self):
		return self._background_frequencies_source
	
	@property
	def strands(self):
		return self._strands

	def add_motif(self, motif: MEMEWriterMotif) -> None:
		self._motifs.append(motif)

	def write(self, file_path: PathLike) -> None:

		output = ""
		output += f"MEME version {self._memesuite_version}\n\n"
		if self._alphabet:
			output += f"ALPHABET= {self._alphabet}\n\n"
		if self._strands:
			output += f"strands: {self._strands}\n\n"
		if self._background_frequencies:
			output += f"Background letter frequencies"
			if self._background_frequencies_source:
				output += f" (from {self._background_frequencies_source}):"
			output += "\n"
			output += f"{self._background_frequencies}\n\n"
		for motif in self._motifs:
			output += str(motif)
			output += "\n\n"
		try:
			# Open the file in write mode
			with open(file_path, "w") as file:
				# Write the string to the file
				file.write(output)
		except IOError:
			print(f"An error occurred while writing to the file {file_path}")
	

	def __repr__(self) -> str:
		return f"MEMEWriter(memesuite_version={self._memesuite_version}, motifs={self._motifs}, alphabet={self._alphabet}, background_frequencies={self._background_frequencies}, strands={self._strands})"


def array_to_string(array: np.ndarray, precision: int) -> str:
	"""Convert a 2D numpy array to a string with the given precision."""
	# create a format string with the desired precision
	float_formatter = "{:." + str(precision) + "f}"
	
	# manually format each float in the array and join them with spaces
	string_rows = [" ".join(float_formatter.format(x) for x in row) for row in array]

	# join rows with line breaks to retain 2D structure
	return "\n".join(string_rows)
