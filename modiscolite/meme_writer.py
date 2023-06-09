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
		self.name = name
		self.probability_matrix = probability_matrix
		self.source_sites = source_sites
		self.alphabet_length = len(alphabet) if alphabet_length is None else alphabet_length
		self.e_value = e_value
		self.url = url

	def __repr__(self) -> str:
		return f"MEMEWriterMotif(name={self.name})"

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
		self.memesuite_version = memesuite_version
		self.motifs = motifs if motifs is not None else []
		self.alphabet = alphabet
		self.background_frequencies = background_frequencies
		self.background_frequencies_source = background_frequencies_source
		self.strands = strands

	def add_motif(self, motif: MEMEWriterMotif) -> None:
		self.motifs.append(motif)

	def get_output(self) -> str:
		output = ""
		output += f"MEME version {self.memesuite_version}\n\n"
		if self.alphabet:
			output += f"ALPHABET= {self.alphabet}\n\n"
		if self.strands:
			output += f"strands: {self.strands}\n\n"
		if self.background_frequencies:
			output += f"Background letter frequencies"
			if self.background_frequencies_source:
				output += f" (from {self.background_frequencies_source}):"
			output += "\n"
			output += f"{self.background_frequencies}\n\n"
		for motif in self.motifs:
			output += str(motif)
			output += "\n\n"
		return output

	def write(self, file_path: PathLike) -> None:
		try:
			# Open the file in write mode
			with open(file_path, "w") as file:
				# Write the string to the file
				file.write(self.get_output())
		except IOError:
			raise IOError(f"MEMEWriter: Could not write to file {file_path}")
	
	def __repr__(self) -> str:
		return f"MEMEWriter(memesuite_version={self.memesuite_version}, motifs={self.motifs}, alphabet={self.alphabet}, background_frequencies={self.background_frequencies}, strands={self.strands})"


def array_to_string(array: np.ndarray, precision: int) -> str:
	"""Convert a 2D numpy array to a string with the given precision."""
	# create a format string with the desired precision
	float_formatter = "{:." + str(precision) + "f}"
	
	# manually format each float in the array and join them with spaces
	string_rows = [" ".join(float_formatter.format(x) for x in row) for row in array]

	# join rows with line breaks to retain 2D structure
	return "\n".join(string_rows)
