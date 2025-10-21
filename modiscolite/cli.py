# tf-modisco command-line tool
# Author: Jacob Schreiber <jmschreiber91@gmail.com>, Ivy Raine <ivy.ember.raine@gmail.com>

from pathlib import Path
from typing import List, Literal, Union, Optional

import click
import h5py
import numpy as np

import modiscolite
from modiscolite.util import calculate_window_offsets, MemeDataType


def _split_chroms(chroms: str) -> Union[List[str], Literal["*"]]:
    """Convert comma-delimited ``chroms`` argument to a list or '*' literal."""
    return "*" if chroms == "*" else chroms.split(",")


@click.group(
    help="""TF-MoDISco is a motif detection algorithm that takes in nucleotide
sequence and their neural-network attribution scores, then extracts motifs that
are repeatedly enriched for attribution signal across the dataset. Use the
sub-commands below to run motif discovery, generate reports, or convert result
files between formats."""
)
def cli() -> None:
    pass


@cli.command(help="Run TF-MoDISco and extract the motifs.")
@click.option(
    "-s",
    "--sequences",
    'seq_path',
    type=click.Path(exists=True),
    help="A .npy or .npz file containing the one-hot encoded sequences.",
)
@click.option(
    "-a",
    "--attributions",
    'attr_path',
    type=click.Path(exists=True),
    help="A .npy or .npz file containing the hypothetical attributions, i.e., the attributions for all nucleotides at all positions."
)
@click.option(
    "-i",
    "--h5py",
    'h5_path',
    type=click.Path(exists=True),
    help="Legacy HDF5 file that stores both sequences and attribution scores.",
)
@click.option(
    "-n",
    "--max-seqlets",
    type=int,
    required=True,
    help="The maximum number of seqlets per metacluster."
)
@click.option(
    "-l",
    "--n-leiden",
    type=int,
    default=50,
    show_default=True,
    help="The number of Leiden clusterings to perform with different random seeds."
)
@click.option(
    "-w",
    "--window",
    type=int,
    default=400,
    show_default=True,
    help="The window surrounding the peak center that will be considered for motif discovery."
)
@click.option(
    "-z",
    "--size",
    "sliding",
    type=int,
    default=20,
    show_default=True,
    help="The size of the seqlet cores, corresponding to `sliding_window_size`."
)
@click.option(
    "-t",
    "--trim-size",
    type=int,
    default=30,
    show_default=True,
    help="The size to trim to, corresponding to `trim_to_window_size`."
)
@click.option(
    "-f",
    "--seqlet-flank-size",
    type=int,
    default=5,
    show_default=True,
    help="The size of the flanks to add to each seqlet, corresponding to `flank_size`."
)
@click.option(
    "-g",
    "--initial-flank-to-add",
    type=int,
    default=10,
    show_default=True,
    help="The size of the flanks to add to each pattern initially, corresponding to `initial_flank_to_add`."
)
@click.option(
    "-j",
    "--final-flank-to-add",
    type=int,
    default=0,
    show_default=True,
    help="The size of the flanks to add to each pattern at the end, corresponding to `final_flank_to_add`."
)
@click.option(
    "-o",
    "--output",
    type=click.Path(),
    default="modisco_results.h5",
    show_default=True,
    help="Path to the output HDF5 file.",
)
@click.option("-v", "--verbose", is_flag=True)
def motifs(
    seq_path: str,
    attr_path: str,
    h5_path: str,
    max_seqlets: int,
    n_leiden: int,
    window: int,
    sliding: int,
    trim_size: int,
    seqlet_flank_size: int,
    initial_flank_to_add: int,
    final_flank_to_add: int,
    output: str,
    verbose: bool,
):
    """Run TF-MoDISco and extract the motifs."""
    if h5_path:
        f = h5py.File(h5_path, "r")
        try:
            center = f["hyp_scores"].shape[1] // 2
            start, end = calculate_window_offsets(center, window)
            attributions = f["hyp_scores"][..., :][..., start:end, :]
            sequences = f["input_seqs"][..., :][..., start:end, :]
        except KeyError:
            center = f["shap"]["seq"].shape[2] // 2
            start, end = calculate_window_offsets(center, window)
            attributions = f["shap"]["seq"][..., :, start:end].transpose(0, 2, 1)
            sequences = f["raw"]["seq"][..., :, start:end].transpose(0, 2, 1)
        f.close()
    else:
        seq_path = Path(seq_path)
        attr_path = Path(attr_path)
        sequences = (
            np.load(seq_path)["arr_0"]
            if seq_path.suffix == ".npz"
            else np.load(seq_path)
        )
        attributions = (
            np.load(attr_path)["arr_0"]
            if attr_path.suffix == ".npz"
            else np.load(attr_path)
        )
        center = sequences.shape[2] // 2
        start, end = calculate_window_offsets(center, window)
        sequences = sequences[:, :, start:end].transpose(0, 2, 1)
        attributions = attributions[:, :, start:end].transpose(0, 2, 1)

    if sequences.shape[1] < window:
        raise ValueError(
            f"Window ({window}) cannot be longer than the sequence length ({sequences.shape[1]})."
        )

    pos_patterns, neg_patterns = modiscolite.tfmodisco.TFMoDISco(
        one_hot=sequences.astype("float32"),
        hypothetical_contribs=attributions.astype("float32"),
        max_seqlets_per_metacluster=max_seqlets,
        sliding_window_size=sliding,
        flank_size=seqlet_flank_size,
        trim_to_window_size=trim_size,
        initial_flank_to_add=initial_flank_to_add,
        final_flank_to_add=final_flank_to_add,
        target_seqlet_fdr=0.05,
        n_leiden_runs=n_leiden,
        verbose=verbose,
    )
    modiscolite.io.save_hdf5(output, pos_patterns, neg_patterns, window)


@cli.command(
    help="Create an HTML report (logos + optional TOMTOM tables) from an HDF5 results file."
)
@click.option(
    "-i",
    "--h5py",
    "h5_path",
    type=click.Path(exists=True),
    required=True,
    help="HDF5 file containing the output from modiscolite.",
)
@click.option(
    "-o",
    "--output",
    type=click.Path(),
    required=True,
    help="Directory to write the HTML report and associated assets.",
)
@click.option(
    "-t",
    "--write-tomtom",
    is_flag=True,
    help="Write the TOMTOM results to the output directory if flag is given.",
)
@click.option(
    "-s",
    "--suffix",
    default="./",
    show_default=True,
    help="The suffix to add to the beginning of images. Should be equal to the output if using a Jupyter notebook.",
)
@click.option(
    "-m",
    "--meme-db",
    type=click.Path(exists=True),
    default=None,
    help="A MEME file containing motifs.",
)
@click.option(
    "-n",
    "--n-matches",
    default=3,
    type=int,
    show_default=True,
    help="The number of top TOMTOM matches to include in the report.",
)
@click.option(
    "-l",
    "--lite",
    is_flag=True,
    help="Whether to use tomtom-lite when mapping patterns to motifs. Note that this also changes the distance function from correlation to Euclidean distance, and so the best motif may differ when there are many similar versions.",
)
def report(
    h5_path, output, write_tomtom, suffix, meme_db, n_matches, lite
):
    """Generate an interactive HTML motif report."""
    modiscolite.report.report_motifs(
        h5_path,
        output,
        img_path_suffix=suffix,
        meme_motif_db=meme_db,
        is_writing_tomtom_matrix=write_tomtom,
        top_n_matches=n_matches,
        ttl=lite,
    )


@cli.command(help="Convert an old HDF5 file to the new format.")
@click.option(
    "-i",
    "--h5py",
    "h5_path",
    type=click.Path(exists=True),
    required=True,
    help="An HDF5 file formatted in the old way.",
)
@click.option(
    "-o",
    "--output",
    type=click.Path(),
    required=True,
    help="An HDF5 file formatted in the new way.",
)
def convert(h5_path, output):
    """Convert old HDF5 to new format."""
    modiscolite.io.convert(h5_path, output)


@cli.command(help="Convert a new HDF5 file back to the legacy format.")
@click.option(
    "-i",
    "--h5py",
    "h5_path",
    type=click.Path(exists=True),
    required=True,
    help="An HDF5 file formatted in the new way.",
)
@click.option(
    "-o",
    "--output",
    type=click.Path(),
    required=True,
    help="An HDF5 file formatted in the old way.",
)
def convert_backward(h5_path, output):
    """Convert new HDF5 to original legacy format."""
    modiscolite.io.convert_new_to_old(h5_path, output)


@cli.command(help="Write a MEME file from a results HDF5.")
@click.option(
    "-i",
    "--h5py",
    "h5_path",
    type=click.Path(exists=True),
    help="An HDF5 file containing the output from modiscolite.",
)
@click.option(
    "-t",
    "--datatype",
    type=MemeDataType,
    required=True,
    help="""A case-sensitive string specifying the desired data of the output file.,
The options are as follows:
- 'PFM':      The position-frequency matrix.
- 'CWM':      The contribution-weight matrix.
- 'hCWM':     The hypothetical contribution-weight matrix; hypothetical
              contribution scores are the contributions of nucleotides not encoded
              by the one-hot encoding sequence. 
- 'CWM-PFM':  The softmax of the contribution-weight matrix.
- 'hCWM-PFM': The softmax of the hypothetical contribution-weight matrix."""
)
@click.option(
    "-o",
    "--output",
    type=click.Path(),
    default=None,
    help="The path to the output file.",
)
@click.option("-q", "--quiet", is_flag=True, help="Suppress output to stdout.")
def meme(h5_path, datatype, output, quiet):
    modiscolite.io.write_meme_from_h5(h5_path, datatype, output, quiet)


@cli.command(help="Output a BED file of seqlets from a modisco results file to stdout (default) and/or to a file (if specified).")
@click.option(
    "-i",
    "--h5py",
    "h5_path",
    type=click.Path(exists=True),
    required=True,
    help="An HDF5 file containing the output from modiscolite.",
)
@click.option(
    "-o",
    "--output",
    type=click.Path(),
    default=None,
    help="The path to the output file.",
)
@click.option(
    "-p",
    "--peaksfile",
    type=click.Path(exists=True),
    required=True,
    help="The path to the peaks file. This is to compute the absolute start and\
end positions of the seqlets within a reference genome, as well as the chroms.",
)
@click.option(
    "-c",
    "--chroms",
    callback=lambda _, __, val: _split_chroms(val),
    required=True,
    help="""A comma-delimited list of chromosomes, or '*', denoting which
chromosomes to process. Should be the same set of chromosomes used during
interpretation. '*' will use every chr in the provided peaks file.
Examples: 'chr1,chr2,chrX' || '*' || '1,2,X'.""",
)
@click.option("-q", "--quiet", is_flag=True, help="Suppress output to stdout.")
@click.option(
    "-w",
    "--windowsize",
    type=int,
    default=None,
    help="""Optional. This is for backwards compatibility for older modisco h5
files that don't contain the window size as an attribute. This should be set
the to size of the window around the peak center that was used for.""",
)
def seqlet_bed(
    h5_path: str,
    output: Optional[str],
    peaksfile: str,
    chroms: Union[List[str], Literal["*"]],
    quiet: bool,
    windowsize: Optional[int],
) -> None:
    """Output a BED file of seqlets from a modisco results file."""
    modiscolite.io.write_bed_from_h5(
        h5_path, peaksfile, output, chroms, windowsize, quiet
    )


@cli.command(help="Output a FASTA file of seqlets from a modisco results file to stdout (default) and/or to a file (if specified).")
@click.option(
    "-i",
    "--h5py",
    "h5_path",
    type=click.Path(exists=True),
    required=True,
    help="An HDF5 file containing the output from modiscolite.",
)
@click.option(
    "-o",
    "--output",
    type=click.Path(),
    default=None,
    help="The path to the output file.",
)
@click.option(
    "-p",
    "--peaksfile",
    type=click.Path(exists=True),
    required=True,
    help="The path to the peaks file. This is to compute the absolute start and\
end positions of the seqlets within a reference genome, as well as the chroms.",
)
@click.option(
    "-s",
    "--sequences",
    type=click.Path(exists=True),
    required=True,
    help="A .npy or .npz file containing the one-hot encoded sequences.",
)
@click.option(
    "-c",
    "--chroms",
    callback=lambda _, __, val: _split_chroms(val),
    required=True,
    help="""A comma-delimited list of chromosomes, or '*', denoting which
chromosomes to process. Should be the same set of chromosomes used during
interpretation. '*' will use every chr in the provided peaks file.
Examples: 'chr1,chr2,chrX' || '*' || '1,2,X'.""",
)
@click.option("-q", "--quiet", is_flag=True, help="Suppress output to stdout.")
@click.option(
    "-w",
    "--windowsize",
    type=int,
    default=None,
    help="""Optional. This is for backwards compatibility for older modisco h5
files that don't contain the window size as an attribute. This should be set
the to size of the window around the peak center that was used for.""",
)
def seqlet_fasta(
    h5_path: str,
    output: Optional[str],
    peaksfile: str,
    sequences: str,
    chroms: Union[List[str], Literal["*"]],
    quiet: bool,
    windowsize: Optional[int],
) -> None:
    modiscolite.io.write_fasta_from_h5(
        h5_path, peaksfile, sequences, output, chroms, windowsize, quiet
    )