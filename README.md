# tfmodisco-lite

tfmodisco is a biological motif discovery algorithm that differentiates itself by using attribution scores from a machine learning model, in addition to the sequences themselves, to guide the discovery. Although in many of our examples this model is [BPNet](https://www.nature.com/articles/s41588-021-00782-6) and the attributions are [DeepLIFT/DeepSHAP](https://captum.ai/api/deep_lift_shap.html) scores, there is no limit on what attribution algorithm is used, or what model the attributions come from. This means that, for example, one would use [attributions from a gapped k-mer SVM](https://academic.oup.com/bioinformatics/article/35/14/i173/5529147?login=false) just as easily as [DeepSHAP on a convolutional neural network that predicts enhancer activity](https://www.nature.com/articles/s41588-022-01048-5). All that's needed to run TF-MoDISco are sequences and their corresponding per-position attribution scores.

The outputs of TF-MoDISco are motifs that summarize repeated patterns with high attribution. These patterns can be visualized using the `reports.report_motifs` function to generate an HTML file (which can be loaded into a Jupyter notebook) that, after training a BPNet model on a SPI1 data set, looks like the following:  

![image](https://user-images.githubusercontent.com/3916816/189726765-47e043c5-c942-4547-9b69-bfc8b5ba3131.png)

tfmodisco-lite is a lightweight version of the original [tfmodisco](https://github.com/kundajelab/tfmodisco) implementation, which takes significantly less memory, (sometimes) runs faster, and is significantly less code, making it easier to understand. This rewrite is an exact reimplementation (except for one minor fix) and so should be able to be used as a drop-in replacement for existing tfmodisco pipelines. 

### Installation

You can install tfmodisco-lite using `pip install modisco-lite`

### Command Line Tools

You can run tfmodisco-lite using the command line tool `modisco` which comes with the tfmodisco-lite installation. This tool allows you to run tfmodisco-lite on a set of sequences and corresponding attributions, and then to generate a report (like the one seen above) for the output generated from the first step.

#### Running tfmodisco-lite

`modisco motifs -s ohe.npz -a shap.npz -n 2000 -o modisco_results.h5`

This command will run modisco on the one-hot encoded sequences in `ohe.npz`, use the attributions from `shap.npz`, use a maximum of 2000 seqlets per metacluster (this is low, but a good starting point for testing the algorithm on your own data), and will output the results to `modisco_results.h5`. Note that you can also use `npy` files if you don't want to use compressed data for some reason. The output saved in `modisco_results.h5` will include all of the patterns, all of the seqlets found per pattern, and a large amount of information about the run.

#### Generating reports

`modisco report -i modisco_results.h5 -o report/ -m motifs.txt -l motifs/pfms/`

This command will take the results from the tfmodisco-lite run, as well as a reference database of motifs to compare the extracted patterns to, and generate a HTML report like the one seen above. Each pattern that is extracted by tfmodisco-lite is compared against the database of motifs using [TOMTOM](https://meme-suite.org/meme/tools/tomtom) to match them with prior knowledge. Note that, for now, you have to pass in a text file containing all the motifs as well as a directory of extracted pfms. This will change in the future. 
