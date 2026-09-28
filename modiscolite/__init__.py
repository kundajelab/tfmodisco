import importlib

from . import aggregator
from . import affinitymat
from . import extract_seqlets
from . import core
from . import gapped_kmer
from . import io
from . import util
from . import tfmodisco
from . import cluster
from . import meme_writer
from . import bed_writer
from . import fasta_writer

__version__ = '2.5.2'


# The report modules load matplotlib, pandas, logomaker, memelite and jinja2,
# which only the report commands use, so they are imported on first access
# (modiscolite.report, from modiscolite import descriptive_report).
_LAZY = ('report', 'descriptive_report')


def __getattr__(name):
	if name in _LAZY:
		return importlib.import_module('.' + name, __name__)
	raise AttributeError('module {!r} has no attribute {!r}'.format(__name__,
		name))
