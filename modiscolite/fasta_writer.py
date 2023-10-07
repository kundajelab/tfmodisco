# Implements a simple FASTAWriter class, which writes a FASTA file from a
# pairs of header and sequence.

# Authors: Ivy Raine <ivy.ember.raine@gmail.com>, Jacob Schreiber <jmschreiber91@gmail.com>

from os import PathLike
from typing import List, Optional


class FASTAEntry:
	"""Contains a sequence and its header pair.

	Parameters
	----------
	header : str
		The header of the sequence.
	sequence : str
		The sequence.
	"""

	def __init__(self, header: str, sequence: str) -> None:
		self.header = header
		self.sequence = sequence
	
	def __repr__(self) -> str:
		return f"FASTAEntry(header={self.header}, sequence={self.sequence})"
	
	def __str__(self) -> str:
		return f">{self.header}\n{self.sequence}"


class FASTAWriter:
	"""Writes pairs of header and sequence to a FASTA file. 

	Parameters
	----------
	pairs : Optional[List[FASTAEntry]]
		The pairs of header and sequence to write to the FASTA file.
	"""
	
	def __init__(self, entries: Optional[List[FASTAEntry]] = []) -> None:
		self.entries = []

	def __repr__(self) -> str:
		return f"FASTAWriter(entries={self.entries})"
	
	def __str__(self) -> str:
		return "\n".join([str(entry) for entry in self.entries])
	
	def add_pair(self, pair: FASTAEntry) -> None:
		"""Add a pair to the FASTAWriter.
		
		Parameters
		----------
		pair : FASTAEntry
			The pair to add to the FASTAWriter.
		"""
		self.entries.append(pair)

	def get_output(self) -> str:
		"""Get the FASTA file as a string.
		
		Returns
		-------
		str
			The FASTA file as a string.
		"""
		return str(self)

	def write(self, path: PathLike) -> None:
		"""Write the FASTA file to the given path.
		
		Parameters
		----------
		path : PathLike
			The path to write the FASTA file to.
		"""
		try:
			with open(path, "w") as f:
				f.write(self.get_output())
		except IOError:
			raise IOError(f"FASTAWriter: Could not write to file {path}.")
