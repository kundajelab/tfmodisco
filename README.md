# TF-MoDISco

TF-MoDISco (**T**ranscription **F**actor **Mo**tif **D**iscovery from **I**mportance **Sco**res) is an algorithm for discovering sequence motifs from machine-learning-model-derived importance scores. Unlike traditional motif discovery methods that rely solely on sequence enrichment, TF-MoDISco leverages context-aware importance scores to identify patterns.

These importance scores can be generated using various attribution methods, such as DeepLIFT or SHAP, applied to models like BPNet. The algorithm identifies high-importance regions (seqlets), clusters them into motifs, and provides a report comparing discovered motifs to known databases.

> [!IMPORTANT]  
> Starting from version v2, TF-MoDISco utilizes the [tfmodisco-lite](https://github.com/jmschrei/tfmodisco-lite/) implementation and interface. This implementation is significantly more memory efficient, and in many cases faster, than the original implementation. The original implementation (v0) is still available [here](https://github.com/kundajelab/tfmodisco/tree/v0-final).

## Algorithm Description

The TF-MoDISco algorithm starts with a set of importance scores on genomic sequences and performs the following tasks:

1. Identify high-importance windows of the sequences, termed "seqlets"
2. Divide the seqlets into positive and negative sets (metaclusters) based on the overall importance score of each seqlet
3. Cluster recurring similar seqlets
4. Generate motifs by aligning the clustered seqlets

During clustering, a coarse-grained similarity is calculated as the cosine similarity between gapped k-mer representations between all pairs of seqlets. This information is used to calculate the top nearest neighbors, for which a fine-grained similarity is calculated as the maximum Jaccard index as two seqlets are aligned with all possible offsets. This sparse similarity matrix is then density adapted, similarly to t-SNE, and Leiden clustering is used to extract patterns. Finally, some heuristics are used to merge similar patterns and split apart the seqlets comprising dissimilar ones.

![image](assets/overview.svg)

## References

TF-MoDISco is described in:
> Wang, Tseng, Ramalingam, Schreiber, et al. "Decoding predictive motif lexicons and syntax from deep learning models of transcription factor binding profiles." (manuscript in preparation)

Related tools:
- [Fi-NeMo](https://github.com/kundajelab/Fi-NeMo): Motif instance detection using TF-MoDISco patterns
- [BPNet](https://github.com/kundajelab/bpnet-refactor): Deep learning models for TF binding prediction
- [ChromBPNet](https://github.com/kundajelab/chrombpnet): Deep learning models for chromatin accessibility prediction

## Installation

You can install TF-MoDISco using `pip install modisco`

## Running TF-MoDISco (test)

You can run TF-MoDISco using the command line tool `modisco` which comes with the TF-MoDISco installation. This tool allows you to run TF-MoDISco on a set of sequences and corresponding attributions, and then to generate a report (like the one seen above) for the output generated from the first step.

`modisco motifs -s ohe.npz -a shap.npz -n 2000 -o modisco_results.h5`

This command will run modisco on the one-hot encoded sequences in `ohe.npz`, use the attributions from `shap.npz`, use a maximum of 2000 positive/negative seqlets (this is low, but a good starting point for testing the algorithm on your own data), and will output the results to `modisco_results.h5`. The one-hot encoded sequences and attributions are assumed to be in length-last format, i.e., have the shape (# examples, 4, sequence length). Note that you can also use `npy` files if you don't want to use compressed data for some reason. 

> [!TIP]
> **Window size:** By default, TF-MoDISco uses a window size of 400 around the center of each input region. You can override this default with `-w`.
>
> **Max seqlets:** Seqlets will generally follow the order of the input regions, and hence can be biased by the order in which the regions are provided. `-n` takes top seqlets in order that they are identified, where identification occurs per region (in order that they are inputted), then in desending order of each seqlet's attribution score per region. For unbiased sampling, shuffle the input regions beforehand. Keep the shuffled regions to keep track of the absolute instance positions.

The output saved in `modisco_results.h5` will include all of the patterns and has the following struture:

```
pos_patterns/
    pattern_0/
        sequence: [...]
        contrib_scores: [...]
        hypothetical_contribs: [...]
        seqlets/
            n_seqlets: [...]
            start: [...]
            end: [...]
            example_idx: [...]
            is_revcomp: [...]
            sequence: [...]
            contrib_scores: [...]
            hypothetical_contribs: [...]
        subpattern_0/
            ...
    pattern_1/
        ...
    ...
neg_patterns/
    pattern_0/
        ...
    pattern_1/
        ...
    ...
```

where `[...]` denotes that data is stored at that attribute. Importantly, the seqlets are all in the correct orientation. If a seqlet has been flipped to be the reverse complement, the sequence, contribution scores, and coordinates have also been flipped. In cases where there are not enough seqlets to consider a metacluster, that attribute (`neg_patterns` or `pos_patterns`) may not appear in the file.

## Generating reports

The TF-MoDISco report can be generated with the following command:
```sh
modisco report -i modisco_results.h5 -o report/ -s report/ -m motifs.txt
```

Each pattern produced by TF-MoDISco is compared against the database of motifs using [TOMTOM](https://meme-suite.org/meme/tools/tomtom). A good default choice is [this collection of human motifs](https://raw.githubusercontent.com/kundajelab/MotifCompendium/refs/heads/main/pipeline/data/MotifCompendium-Database-Human.meme.txt) produced by the [MotifCompendium](https://github.com/kundajelab/MotifCompendium) package.

The report details each pattern, including seqlet importance and spatial distributions, example seqlets at different importance levels, and motif visualizations.

For users who need the legacy report format use:
```sh
modisco report-simple -i modisco_results.h5 -o simple_report/ -s simple_report/ -m motifs.txt
```

## Custom alphabets (proteins, RNA, ...)

TF-MoDISco can run on any alphabet whose length matches the trailing dimension
of the one-hot encoding. This is exposed via an `alphabet` keyword on
`TFMoDISco(...)` and `core.TrackSet(...)` (Python API only — the CLI is still
DNA-only). The default is `'ACGT'`, so existing DNA call sites are unchanged.

Examples:
- DNA (default): `alphabet='ACGT'`, one-hot shape `(N, L, 4)`
- RNA: `alphabet='ACGU'`, one-hot shape `(N, L, 4)`
- Protein (20 amino acids): `alphabet='ACDEFGHIKLMNPQRSTVWY'`, one-hot shape `(N, L, 20)`
- Any reduced or custom alphabet whose length equals `one_hot.shape[-1]`

```python
import numpy as np
from modiscolite.tfmodisco import TFMoDISco
from modiscolite.io import save_hdf5

ALPHABET = "ACDEFGHIKLMNPQRSTVWY"   # 20 amino acids
# one_hot: (N, L, 20) float32, hypothetical_contribs: same shape
pos, neg = TFMoDISco(
    one_hot=one_hot,
    hypothetical_contribs=hypothetical_contribs,
    alphabet=ALPHABET,
    sliding_window_size=6, flank_size=2,
    trim_to_window_size=4, initial_flank_to_add=2,
    min_metacluster_size=20, max_seqlets_per_metacluster=4000,
    final_min_cluster_size=15,
    min_ic_in_window=0.3, min_ic_windowsize=3,
    target_seqlet_fdr=0.2,
)
save_hdf5("modisco.h5", pos, neg, window_size=6)
```

Behavior changes when `alphabet != 'ACGT'`:
- The reverse-complement augmentation in `aggregator._align_patterns` is
  auto-disabled. RC is only meaningful for DNA-like complementary alphabets;
  for any other alphabet, patterns are aligned in the forward direction only.
- `save_hdf5` stamps the alphabet as an HDF5 root attribute (`alphabet`).
  Older h5 files without this attribute are read as DNA, preserving back-compat.
- `modisco report` writes a `modisco_cwm_rev` column for every pattern. For
  non-DNA alphabets that column is a **length-reversed** (C → N read) view of
  the forward CWM, not a biological reverse complement. The HTML report
  includes a banner noting this.
- The MEME writer drops the `strands: + -` line and emits an alphabet block
  matching the alphabet used. The motif logo uses a 20-color protein scheme
  when `len(alphabet) >= 20` and the default WebLogo NA scheme otherwise.

Tips for protein ISM scores: pass mean-centered contributions as
`hypothetical_contribs = deltas - deltas.mean(axis=-1, keepdims=True)` so the
WT-AA channel carries a positive contribution at residues where mutations are
costly. Tune `sliding_window_size`, `trim_to_window_size`, `min_ic_in_window`,
`min_ic_windowsize`, and `min_metacluster_size` for short motifs; the DNA
defaults assume ~20 bp windows.
