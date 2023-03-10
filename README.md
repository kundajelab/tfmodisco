# tfmodisco-lite

> **Warning**
> tfmodisco-lite v2.0.0 and above may produce slightly different results from the original TF-MoDISCo code as minor bugs are fixed and some speed improvements required swapping sorting algorithms.

TF-MoDISco is a biological motif discovery algorithm that differentiates itself by using attribution scores from a machine learning model, in addition to the sequence itself, to guide motif discovery. Using the attribution scores, as opposed to the signal being predicted by the machine learning model (e.g. ChIP-seq peaks), can be beneficial because the attributions fine-map the specific sequence drivers of biology. Although in many of our examples this model is [BPNet](https://www.nature.com/articles/s41588-021-00782-6) and the attributions are from [DeepLIFT/DeepSHAP](https://captum.ai/api/deep_lift_shap.html), there is no limit on what attribution algorithm is used, or what model the attributions come from. This means that, for example, one would use [attributions from a gapped k-mer SVM](https://academic.oup.com/bioinformatics/article/35/14/i173/5529147?login=false) just as easily as [DeepSHAP on a convolutional neural network that predicts enhancer activity](https://www.nature.com/articles/s41588-022-01048-5). All that's needed to run TF-MoDISco are sequences and their corresponding per-position attribution scores.

> **Note**
> This is a rewrite of the original TF-MoDISCo code, which can be found at https://github.com/kundajelab/tfmodisco

Below, we can see just how fast and memory efficient tfmodisco-lite is on a an ATAC-seq experiment when capped at 100k seqlets. The original tfmodisco code (left) takes 5 hours and 35 Gb of RAM. tfmodisco-lite v1.0.0 (middle) takes closer to 4 hours and only requires 13 Gb of RAM. tfmodisco-lite v2.0.0 (right) takes only a little over 1.5 hours and 8 Gb of RAM.

![image](https://user-images.githubusercontent.com/3916816/224192946-43434221-6da1-4875-ab00-6f782f9178ae.png)


### Installation

You can install tfmodisco-lite using `pip install modisco-lite`

### Command Line Tools

You can run tfmodisco-lite using the command line tool `modisco` which comes with the tfmodisco-lite installation. This tool allows you to run tfmodisco-lite on a set of sequences and corresponding attributions, and then to generate a report (like the one seen above) for the output generated from the first step.

### Algorithm Description

At a high level, the procedure works like this: "seqlets," which are short spans of sequence with high absolute attribution score, are extracted from the given examples. These seqlets are then divided into positive and negative seqlets ("metaclusters"). For each metacluster, a coarse-grained similarity is calculated as the cosine similarity between gapped k-mer representations between all pairs of seqlets. This information is used to calculate the top nearest neighbors, for which a fine-grained similarity is calculated as the maximum Jaccard index as two seqlets are aligned with all possible offsets. This sparse smilarity matrix is then density adapted, similarly to t-SNE, and Leiden clustering is used to extract patterns. Finally, some heuristics are used to merge similar patterns and split apart the seqlets comprising dissimilar ones. 

The outputs of TF-MoDISco are motifs that summarize repeated patterns with high attribution. These patterns can be visualized using the `reports.report_motifs` function to generate an HTML file (which can be loaded into a Jupyter notebook) that, after training a BPNet model on a SPI1 data set, looks like the following:  

![image](https://user-images.githubusercontent.com/3916816/189726765-47e043c5-c942-4547-9b69-bfc8b5ba3131.png)

tfmodisco-lite is a lightweight version of the original [TF-MoDISco](https://github.com/kundajelab/tfmodisco) implementation, which takes significantly less memory, (sometimes) runs faster, and is significantly less code, making it easier to understand. This rewrite is an exact reimplementation (except for one minor fix) and so should be able to be used as a drop-in replacement for existing tfmodisco pipelines. 

#### Running tfmodisco-lite

`modisco motifs -s ohe.npz -a shap.npz -n 2000 -o modisco_results.h5`

This command will run modisco on the one-hot encoded sequences in `ohe.npz`, use the attributions from `shap.npz`, use a maximum of 2000 seqlets per metacluster (this is low, but a good starting point for testing the algorithm on your own data), and will output the results to `modisco_results.h5`. The one-hot encoded sequences and attributions are assumed to be in length-last format, i.e., have the shape (# examples, 4, sequence length). Note that you can also use `npy` files if you don't want to use compressed data for some reason. 

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

#### Generating reports

`modisco report -i modisco_results.h5 -o report/ -s report/ -m motifs.txt`

This command will take the results from the tfmodisco-lite run, as well as a reference database of motifs to compare the extracted patterns to, and generate a HTML report like the one seen above. Each pattern that is extracted by tfmodisco-lite is compared against the database of motifs using [TOMTOM](https://meme-suite.org/meme/tools/tomtom) to match them with prior knowledge.
