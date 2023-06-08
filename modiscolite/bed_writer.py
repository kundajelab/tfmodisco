# A class that for writing BED Files, based on the UCSC specification.
# Authors: Ivy Raine <ivy.ember.raine@gmail.com>, Jacob Schreiber <jmschreiber91@gmail.com>
from typing import List, Optional, Literal, Union
from collections import OrderedDict
import numpy as np

from os import PathLike


class BEDTrackLine:
	"""Class for handling Track Line for BED file writing.
	
	Parameters
	----------
	arguments : OrderedDict
		The arguments for the track line.
		Example:
		{
			"name": "my_track",
			"description": "This is my track",
			"visibility": "2",
			"color": "0,0,255",
		}
	"""

	def __init__(
		self,
		arguments: OrderedDict
	) -> None:
		self.arguments = arguments


	def __repr__(self) -> str:
		return f"BEDTrackLine(arguments={self.arguments})"


	def __str__(self) -> str:
		output = "track"
		for key, value in self.arguments.items():
			output += f' {key}="{value}"'
		return output


class BEDRow:
	"""Class for handling Row for BED file writing.

	Parameters
	----------
	chrom : str
		The chromosome number.
	chrom_start : int
		The start position of the row.
	chrom_end : int
		The end position of the row.
	name : Optional[str]
		The name of the row.
	score : Optional[int]
		The score of the row.
	strand : Optional[str]
		The strand of the row.
	thick_start : Optional[int]
		The thick start of the row.
	thick_end : Optional[int]
		The thick end of the row.
	item_rgb : Optional[str]
		The item RGB of the row.
	block_count : Optional[int]
		The block count of the row.
	block_sizes : Optional[str]
		A comma-separated list of the block sizes.
		The number of items in this list should correspond to blockCount.
	block_starts : Optional[str]
		A comma-separated list of block starts.
		All of the blockStart positions should be calculated relative to chromStart.
		The number of items in this list should correspond to blockCount.
	"""


	# Dot represents a missing value in BED files.
	IntOrDot = Union[int, Literal['.']]


	def __init__(
		self,
		chrom: str,
		chrom_start: int,
		chrom_end: int,
		name: Optional[str] = None,
		score: Optional[IntOrDot] = None,
		strand: Optional[str] = None,
		thick_start: Optional[IntOrDot] = None,
		thick_end: Optional[IntOrDot] = None,
		item_rgb: Optional[str] = None,
		block_count: Optional[IntOrDot] = None,
		block_sizes: Optional[str] = None,
		block_starts: Optional[str] = None,
	) -> None:
		self.chrom = chrom
		self.chrom_start = chrom_start
		self.chrom_end = chrom_end
		self.name = name
		self.score = score
		self.strand = strand
		self.thick_start = thick_start
		self.thick_end = thick_end
		self.item_rgb = item_rgb
		self.block_count = block_count
		self.block_sizes = block_sizes
		self.block_starts = block_starts

	
	def __str__(self) -> str:
		output = f"{self.chrom}\t{self.chrom_start}\t{self.chrom_end}"
		if self.name is not None:
			output += f"\t{self.name}"
		if self.score is not None:
			output += f"\t{self.score}"
		if self.strand is not None:
			output += f"\t{self.strand}"
		if self.thick_start is not None:
			output += f"\t{self.thick_start}"
		if self.thick_end is not None:
			output += f"\t{self.thick_end}"
		if self.item_rgb is not None:
			output += f"\t{self.item_rgb}"
		if self.block_count is not None:
			output += f"\t{self.block_count}"
		if self.block_sizes is not None:
			output += f"\t{self.block_sizes}"
		if self.block_starts is not None:
			output += f"\t{self.block_starts}"
		return output


class BEDTrack:
	"""Class for handling Track for BED file writing.

	Parameters
	----------
	track_line : Optional[BEDTrackLine]
		The track line for the track.
	rows : List[BEDRow]
		The rows for the track.

	Usage
	-----
	>>> track = BEDTrack(
	... 	track_line=BEDTrackLine(
	... 		OrderedDict(
	... 			name="my_track",
	... 			description="This is my track",
	... 			visibility="2",
	... 			color="0,0,255",
	... 		)
	... 	),
	... 	rows=[
	... 		BEDRow(
	... 			chrom="chr1",
	... 			chrom_start=100,
	... 			chrom_end=200,
	... 			name="my_row",
	... 			score=100,
	... 			strand="+",
	... 			thick_start=100,
	... 			thick_end=200,
	... 			item_rgb="0,0,255",
	... 			block_count=1,
	... 			block_sizes="100",
	... 			block_starts="0",
	... 		)
	... 	]
	... )
	>>> track.add_row(
	... 	BEDRow(
	... 		chrom="chr1",
	... 		chrom_start=300,
	... 		chrom_end=400,
	... 		name="my_row2",
	... 		score=100,
	... 		strand="+",
	... 		thick_start=300,
	... 		thick_end=400,
	... 		item_rgb="0,0,255",
	... 		block_count=1,
	... 		block_sizes="100",
	... 		block_starts="0",
	... 	)
	... )
	"""
	
	def __init__(
		self,
		track_line: Optional[BEDTrackLine] = None,
		rows: List[BEDRow] = []
	) -> None:
		self.track_line = track_line
		self.rows = rows


	def add_row(self, row: BEDRow) -> None:
		"""Add a row to the track.

		Parameters
		----------
		row : BEDRow
			The row to add to the track.
		"""
		self.rows.append(row)
	

	def __repr__(self) -> str:
		return f"BEDTrack(track_line={self.track_line}, rows={self.rows})"
	

	def __str__(self) -> str:
		output = str(self.track_line) + "\n" if self.track_line is not None else ""
		for row in self.rows:
			output += str(row) + "\n"
		return output


class BEDWriter:
	"""Class for writing BED Files.
	
	Usage:
	>>> writer = BEDWriter()
	>>> writer.add_track(
	... 	BEDTrack(
	... 		track_line=BEDTrackLine(
	... 			OrderedDict(
	... 				[
	... 					("name", "my_track"),
	... 					("description", "This is my track"),
	... 					("visibility", "2"),
	... 					("color", "0,0,255"),
	... 				]
	... 			)
	... 		)
	... 	),
	... 	rows=[
	... 		BEDRow("chr1", 100, 200, name="row1"),
	... 		BEDRow("chr1", 300, 400, name="row2")
	... 	]
	... )
	>>> writer.write("my_bed_file.bed")
	"""

	def __init__(
		self
	) -> None:
		self.tracks = []

	
	def add_track(
		self,
		track: BEDTrack
	) -> None:
		self.tracks.append(track)


	def get_output(
		self
	) -> str:
		"""Get the output string for the BED file.

		Returns
		-------
		str
			The output string for the BED file.
		"""
		return "\n".join([str(track) for track in self.tracks])


	def write(
		self,
		file: PathLike
	) -> None:
		"""Write the BED file to the given file.
		
		Parameters
		----------
		file : PathLike
			The file to write to.
		"""

		try:
			with open(file, 'w') as f:
				f.write(self.get_output())
		except IOError:
			raise IOError(f"BEDWriter: Could not write to file {file}.")

