# Create a class that can be used to write MEME files.
# Authors: Ivy Raine <ivy.ember.raine@gmail.com>, Jacob Schreiber <jmschreiber91@gmail.com>
from typing import List, Optional
import numpy as np

from os import PathLike

class MemeWriterMotif:
	"""Class for handling Motif for MEME file writing."""

	def __init__(
		self,
		name: str,
		probability_matrix: np.ndarray,
		# `alphabet_length` is known as 'alength' in MEME Suite.
		alphabet_length: int,
		# `source_sites` is known as 'nsites' in MEME Suite.
		source_sites: int,
		e_value: Optional[str] = None,
		url: Optional[str] = None
	) -> None:
		self._name = name
		self._probability_matrix = probability_matrix
		self._alphabet_length = alphabet_length
		self._source_sites = source_sites
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
		return f"MemeWriterMotif(name={self._name})"


class MemeWriter:
	"""Class for handling MEME file writing."""

	def __init__(
		self,
		memesuite_version: str,
		# May also be added incrementally using `add_motif()`.
		motifs: List[MemeWriterMotif] = [],
		# Optional (Recommended) by MEME Suite.
		alphabet: Optional[str] = None,
		background_frequencies: Optional[str] = None,
		background_frequencies_source: Optional[str] = None,
		# Optional by MEME Suite.
		strands: Optional[List[str]] = None,
	) -> None:
		self._memesuite_version = memesuite_version
		self._motifs = motifs if motifs is not None else []
		# Alphabet example: "ACGT".
		self._alphabet = alphabet
		# Background frequncies example: "A 0.25 C 0.25 G 0.25 T 0.25".
		self._background_frequencies = background_frequencies
		self._background_frequencies_source = background_frequencies_source
		# Strands example: "+ -".
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

	def add_motif(self, motif: MemeWriterMotif) -> None:
		self._motifs.append(motif)

	def write(self, file_path: PathLike) -> None:

		def array_to_string(array: np.ndarray, precision: int) -> str:
			"""Convert a 2D numpy array to a string with the given precision."""
			# create a format string with the desired precision
			float_formatter = "{:." + str(precision) + "f}"
			
			# manually format each float in the array and join them with spaces
			string_rows = [" ".join(float_formatter.format(x) for x in row) for row in array]

			# join rows with line breaks to retain 2D structure
			return "\n".join(string_rows)

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
			output += f"MOTIF {motif.name}\n"
			output += f"letter-probability matrix: alength= {str(motif.alphabet_length)} w= {motif.probability_matrix.shape[0]} nsites= {str(motif.source_sites)}"
			if motif.e_value:
 				output += f"E= {motif.e_value}"
			output += "\n"
			# Iterate through rows in nparray
			output += array_to_string(motif.probability_matrix, 6)
			output += "\n"
			if motif.url:
				output += f"URL {motif.url}\n"
			output += "\n"
		try:
			# Open the file in write mode
			with open(file_path, "w") as file:
				# Write the string to the file
				file.write(output)
			print(f"Successfully wrote to the file {file_path}.")
		except IOError:
			print(f"An error occurred while writing to the file {file_path}.")
	

	def __repr__(self) -> str:
		return f"MemeWriter(memesuite_version={self._memesuite_version}, motifs={self._motifs}, alphabet={self._alphabet}, background_frequencies={self._background_frequencies}, strands={self._strands})"
