# test_motif.py
# Contact: Jacob Schreiber <jmschreiber91@gmail.com>

import h5py
import pytest

from click.testing import CliRunner

from modiscolite.cli import motifs

from .conftest import DATA_DIR

from numpy.testing import assert_array_equal
from numpy.testing import assert_array_almost_equal


@pytest.mark.cmd
def test_modisco_motif(tmp_path):
	output = tmp_path / "modisco_results.h5"

	runner = CliRunner()
	result = runner.invoke(motifs, ["-s", str(DATA_DIR / "sequences.npz"),
		"-a", str(DATA_DIR / "attributions.npz"), "-n", "150", "-o",
		str(output), "-w", "300", "-v"])

	assert result.exit_code == 0, result.output
	assert result.output == "Using 150 positive seqlets\n"

	with h5py.File(output, "r") as f:
		assert f.attrs["window_size"] == 300
		assert list(f.keys()) == ['pos_patterns']
		assert list(f['pos_patterns'].keys()) == ['pattern_0', 'pattern_1']

		pattern = f['pos_patterns']['pattern_0']
		assert sorted(pattern.keys()) == ['contrib_scores',
			'hypothetical_contribs', 'seqlets', 'sequence', 'subpattern_0',
			'subpattern_1', 'subpattern_2']
		assert_array_equal(pattern['seqlets']['n_seqlets'][:], [77])

		assert_array_almost_equal(pattern['sequence'][:], [
			[0.2468, 0.1818, 0.3247, 0.2468],
			[0.2727, 0.2338, 0.2987, 0.1948],
			[0.2987, 0.2078, 0.3377, 0.1558],
			[0.2078, 0.2857, 0.3117, 0.1948],
			[0.2468, 0.2987, 0.2468, 0.2078],
			[0.2597, 0.2078, 0.3117, 0.2208],
			[0.2468, 0.1948, 0.3766, 0.1818],
			[0.2597, 0.1818, 0.3117, 0.2468],
			[0.2597, 0.2468, 0.1948, 0.2987],
			[0.1818, 0.1948, 0.3247, 0.2987],
			[0.1429, 0.2338, 0.4026, 0.2208],
			[0.2078, 0.3247, 0.3247, 0.1429],
			[0.4156, 0.1558, 0.2468, 0.1818],
			[0.1429, 0.0519, 0.6753, 0.1299],
			[0.0130, 0.0130, 0.9740, 0.0000],
			[0.9870, 0.0000, 0.0130, 0.0000],
			[0.0000, 0.0000, 1.0000, 0.0000],
			[1.0000, 0.0000, 0.0000, 0.0000],
			[0.0000, 0.0000, 1.0000, 0.0000],
			[0.4286, 0.1169, 0.2468, 0.2078],
			[0.0000, 0.0000, 1.0000, 0.0000],
			[0.0000, 0.0000, 1.0000, 0.0000],
			[0.4026, 0.2597, 0.1688, 0.1688],
			[0.3636, 0.1039, 0.3506, 0.1818],
			[0.2338, 0.1688, 0.3506, 0.2468],
			[0.2078, 0.1818, 0.4026, 0.2078],
			[0.2078, 0.1948, 0.3896, 0.2078],
			[0.2857, 0.2597, 0.3117, 0.1429],
			[0.3377, 0.1169, 0.2208, 0.3247],
			[0.3377, 0.1688, 0.2727, 0.2208],
			[0.2987, 0.1558, 0.3247, 0.2208],
			[0.2078, 0.2468, 0.3247, 0.2208],
			[0.2208, 0.2987, 0.2597, 0.2208],
			[0.3636, 0.1429, 0.2078, 0.2857],
			[0.2727, 0.1818, 0.2987, 0.2468],
			[0.2208, 0.2597, 0.2987, 0.2208],
			[0.3117, 0.2857, 0.2468, 0.1558],
			[0.3377, 0.1688, 0.2857, 0.2078],
			[0.1818, 0.2338, 0.2987, 0.2857],
			[0.2078, 0.1688, 0.2857, 0.3377],
			[0.2987, 0.1948, 0.3247, 0.1818],
			[0.2727, 0.2338, 0.3377, 0.1558],
			[0.2727, 0.1299, 0.2987, 0.2987],
			[0.2208, 0.2727, 0.2597, 0.2468],
			[0.2338, 0.2468, 0.2468, 0.2727],
			[0.2078, 0.2468, 0.2338, 0.3117],
			[0.2987, 0.2338, 0.2208, 0.2468],
			[0.3117, 0.2597, 0.2987, 0.1299],
			[0.2468, 0.2857, 0.2468, 0.2208],
			[0.3117, 0.1169, 0.2727, 0.2987]
		], 4)

		pattern = f['pos_patterns']['pattern_1']
		assert sorted(pattern.keys()) == ['contrib_scores',
			'hypothetical_contribs', 'seqlets', 'sequence', 'subpattern_0']
		assert_array_equal(pattern['seqlets']['n_seqlets'][:], [24])
