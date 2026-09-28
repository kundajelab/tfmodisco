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
		assert list(f['pos_patterns'].keys()) == ['pattern_0']

		pattern = f['pos_patterns']['pattern_0']
		assert sorted(pattern.keys()) == ['contrib_scores',
			'hypothetical_contribs', 'seqlets', 'sequence', 'subpattern_0',
			'subpattern_1', 'subpattern_2']
		assert_array_equal(pattern['seqlets']['n_seqlets'][:], [80])

		assert_array_almost_equal(pattern['sequence'][:], [
			[0.2000, 0.2250, 0.2500, 0.3250],
			[0.2375, 0.2000, 0.3250, 0.2375],
			[0.2625, 0.2375, 0.3000, 0.2000],
			[0.2875, 0.2125, 0.3500, 0.1500],
			[0.2125, 0.2875, 0.3125, 0.1875],
			[0.2375, 0.3125, 0.2375, 0.2125],
			[0.2625, 0.2125, 0.3000, 0.2250],
			[0.2375, 0.2125, 0.3750, 0.1750],
			[0.2625, 0.1875, 0.3000, 0.2500],
			[0.2625, 0.2375, 0.2125, 0.2875],
			[0.1750, 0.1875, 0.3500, 0.2875],
			[0.1500, 0.2375, 0.4000, 0.2125],
			[0.2000, 0.3375, 0.3250, 0.1375],
			[0.4375, 0.1500, 0.2375, 0.1750],
			[0.1500, 0.0500, 0.6750, 0.1250],
			[0.0125, 0.0125, 0.9750, 0.0000],
			[0.9875, 0.0000, 0.0125, 0.0000],
			[0.0000, 0.0000, 1.0000, 0.0000],
			[1.0000, 0.0000, 0.0000, 0.0000],
			[0.0000, 0.0000, 1.0000, 0.0000],
			[0.4250, 0.1250, 0.2375, 0.2125],
			[0.0000, 0.0000, 1.0000, 0.0000],
			[0.0000, 0.0000, 1.0000, 0.0000],
			[0.4000, 0.2750, 0.1625, 0.1625],
			[0.3750, 0.1125, 0.3375, 0.1750],
			[0.2375, 0.1750, 0.3500, 0.2375],
			[0.2250, 0.1750, 0.4000, 0.2000],
			[0.2125, 0.2000, 0.3875, 0.2000],
			[0.2875, 0.2500, 0.3125, 0.1500],
			[0.3250, 0.1250, 0.2375, 0.3125],
			[0.3250, 0.1625, 0.2750, 0.2375],
			[0.3000, 0.1500, 0.3250, 0.2250],
			[0.2000, 0.2375, 0.3375, 0.2250],
			[0.2375, 0.3000, 0.2500, 0.2125],
			[0.3625, 0.1375, 0.2125, 0.2875],
			[0.2625, 0.1875, 0.3000, 0.2500],
			[0.2500, 0.2500, 0.2875, 0.2125],
			[0.3125, 0.2750, 0.2625, 0.1500],
			[0.3250, 0.1625, 0.3000, 0.2125],
			[0.1875, 0.2500, 0.2875, 0.2750],
			[0.2250, 0.1625, 0.2875, 0.3250],
			[0.2875, 0.2000, 0.3375, 0.1750],
			[0.2625, 0.2625, 0.3250, 0.1500],
			[0.2875, 0.1375, 0.2875, 0.2875],
			[0.2375, 0.2625, 0.2625, 0.2375],
			[0.2250, 0.2625, 0.2500, 0.2625],
			[0.2125, 0.2500, 0.2375, 0.3000],
			[0.3000, 0.2375, 0.2125, 0.2500],
			[0.3125, 0.2625, 0.2875, 0.1375],
			[0.2500, 0.2750, 0.2500, 0.2250]
		], 4)
