from __future__ import division, absolute_import
import os
import numpy as np
from .. import util
from joblib import Parallel, delayed
from collections import Counter
import os
from subprocess import Popen, PIPE
import time


def run_meme(meme_command, n_jobs, input_file, outdir, nmotifs, revcomp):

    start = time.time()
    #p = Popen([meme_command,input_file,"-dna","-mod","anr",
    #           "-nmotifs",str(nmotifs),
    #           "-minw","6","-maxw","50","-oc",outdir]),
    #          stdout=PIPE, stderr=PIPE, stdin=PIPE)
    #while True:
    #    print("hi")
    #    output = p.stdout.read()
    #    print("ho")
    #    if output == '' and process.poll() is not None:
    #        break
    #    sys.stdout.write(output)
    print("Running MEME")
    command = (meme_command+" "+input_file+" -dna -mod zoops -nmotifs "            
               +str(nmotifs)
               +("" if n_jobs==1 else " -p "+str(n_jobs))
               +" -minw 6 -maxw 50 -objfun classic"
               +(" -revcomp" if revcomp else "")
               +" -markov_order 0 -oc "+outdir)
    print("Command:",command)
    os.system(command)
    print("Duration of MEME:",time.time()-start,"seconds")


class InitClustererFactory(object):

    #need this function simply because the onehot track name might not be
    # known at the time when MemeInitClustererFactory is instantiated
    def set_onehot_track_name(self, onehot_track_name):
        self.onehot_track_name = onehot_track_name


class MemeInitClustererFactory(InitClustererFactory):

    def __init__(self, meme_command, base_outdir, max_num_seqlets_to_use,
                       nmotifs, e_value_threshold=0.05,
                       n_jobs=1, verbose=True):
        self.meme_command = meme_command
        self.base_outdir = base_outdir
        self.max_num_seqlets_to_use = max_num_seqlets_to_use 
        self.nmotifs = nmotifs
        self.call_count = 0 #to avoid overwriting for each metacluster
        self.e_value_threshold = e_value_threshold
        self.n_jobs = n_jobs
        self.verbose = verbose

    def __call__(self, seqlets):

        if (hasattr(self, "onehot_track_name")==False):
            raise RuntimeError("Please call set_onehot_track_name first")

        onehot_track_name = self.onehot_track_name

        outdir = self.base_outdir+"/metacluster"+str(self.call_count)
        self.call_count += 1
        if (os.path.exists(outdir)==False):
            os.makedirs(outdir)

        seqlet_fa_to_write = outdir+"/inp_seqlets.fa"
        seqlet_fa_fh = open(seqlet_fa_to_write, 'w') 
        if (len(seqlets) > self.max_num_seqlets_to_use):
            print(np.random.RandomState(1).choice(     
                         np.arange(self.max_num_seqlets_to_use),                    
                         replace=False))
            seqlets = [seqlets[x] for x in np.random.RandomState(1).choice(
                         np.arange(len(seqlets)),
                         size=self.max_num_seqlets_to_use,
                         replace=False)]

        letter_order = "ACGT"
        for seqlet in seqlets:
            seqlet_fa_fh.write(">"+str(seqlet.coor.example_idx)+":"
                               +str(seqlet.coor.start)+"-"
                               +str(seqlet.coor.end)+"\n") 
            seqlet_onehot = seqlet[onehot_track_name].fwd
            seqlet_fa_fh.write("".join([letter_order[x] for x in
                                np.argmax(seqlet_onehot, axis=-1)])+"\n") 
        seqlet_fa_fh.close()

        #determine whether there's a revcomp sequence
        if (seqlets[0][onehot_track_name].rev is None):
            revcomp = False 
        else:
            revcomp = True

        run_meme(meme_command=self.meme_command,
                 input_file=seqlet_fa_to_write,
                 outdir=outdir, nmotifs=self.nmotifs,
                 n_jobs=self.n_jobs,
                 revcomp=revcomp) 

        motifs = parse_meme(meme_xml=outdir+"/meme.xml",
                            e_value_threshold=self.e_value_threshold)
        return PwmClusterer(
                pwms=motifs, onehot_track_name=self.onehot_track_name,
                n_jobs=self.n_jobs, verbose=self.verbose,
                revcomp=revcomp)


class Pwm(object):

    def __init__(self, matrix, threshold):
        self.matrix = matrix
        self.threshold = threshold
        

def parse_meme(meme_xml, e_value_threshold):
    import xml.etree.ElementTree as ET 
    tree = ET.parse(meme_xml)
    motifs_xml = list(tree.getroot().find("motifs"))
    motifs = []
    motif_pvals = []
    motif_bayesthresholds = []

    for motif_xml in motifs_xml:
        pwm = []
        motif_name = motif_xml.get("name")
        p_value = float(motif_xml.get("p_value"))
        e_value = float(motif_xml.get("e_value"))
        bayes_threshold = float(motif_xml.get("bayes_threshold"))
        alphabet_matrix_xml = list(motif_xml.find("scores")
                                            .find("alphabet_matrix"))
        for pwm_row_xml in alphabet_matrix_xml:
            pwm_row = [float(x.text) for x in list(pwm_row_xml)] 
            pwm.append(pwm_row) 

        if (e_value < e_value_threshold):
            motifs.append(Pwm(matrix=np.array(pwm),
                              threshold=bayes_threshold))
        else:
            print("Skipping motif "+motif_name+" as e-value "+str(e_value)
                  +" does not meet threshold of "+str(e_value_threshold))
    return motifs


def get_max_across_sequences(onehot_seq, weightmat, revcomp):
    fwd_pwm_scan_results = util.compute_pwm_scan(
                                 onehot_seq=onehot_seq, weightmat=weightmat)
    if (revcomp):
        rev_pwm_scan_results = util.compute_pwm_scan(
                                        onehot_seq=onehot_seq,
                                        weightmat=weightmat[::-1, ::-1])
        to_return = np.max(np.maximum(fwd_pwm_scan_results,
                                      rev_pwm_scan_results), axis=-1)
    else:
        to_return = np.max(fwd_pwm_scan_results, axis=-1) 

    return to_return


class PwmClusterer(object):

    def __init__(self, pwms, n_jobs,
                 onehot_track_name,
                 revcomp,
                 verbose=True):
        self.pwms = pwms
        self.n_jobs = n_jobs
        self.revcomp = revcomp
        self.verbose = verbose
        self.onehot_track_name = onehot_track_name

    def __call__(self, seqlets):

        onehot_track_name = self.onehot_track_name

        onehot_seqlets = np.array([x[onehot_track_name].fwd for x in seqlets]) 
        #do a motif scan on onehot_seqlets
        max_pwm_scores_perseq = np.array(Parallel(n_jobs=self.n_jobs)(
                                    delayed(get_max_across_sequences)(
                                     onehot_seqlets, pwm.matrix, self.revcomp)
                                    for pwm in self.pwms))
        #map seqlets to best match motif if min > motif threshold 
        argmax_pwm = np.argmax(max_pwm_scores_perseq, axis=0)
        argmax_pwm_score = np.squeeze(
            np.take_along_axis(max_pwm_scores_perseq,
                               np.expand_dims(argmax_pwm, axis=0),
                               axis=0))

        #seqlet_assigned is a boolean vector indicating whether the seqlet
        # was actually successfully assigned to a cluster
        seqlet_assigned = np.array([True if score > self.pwms[argmax].threshold
                                    else False for argmax,score
                                    in zip(argmax_pwm, argmax_pwm_score)])
        
        #not all pwms may wind up with seqlets assigned to them; if this is
        # the case, then we would want to remap the cluster indices such
        # that every assigned cluster index gets a seqlet assigned to it
        argmax_pwm[seqlet_assigned==False] = -1
        seqlets_per_pwm = Counter(argmax_pwm)
        if (self.verbose):
            print("Of "+str(len(seqlets))+" seqlets, cluster assignments are:",
                  seqlets_per_pwm)
        pwm_cluster_remapping = dict([ (x[1],x[0]) for x in
            enumerate([y for y in sorted(seqlets_per_pwm.keys(),
                                         key=lambda x: -seqlets_per_pwm[x])
                       if seqlets_per_pwm[y] > 0 and y >= 0]) ])

        final_seqlet_clusters = np.zeros(len(seqlets))
        #assign the remapped clusters for the seqlets that received assignment
        final_seqlet_clusters[seqlet_assigned] = np.array(
            [pwm_cluster_remapping[x] for x in argmax_pwm[seqlet_assigned]])
        #for all the unassigned seqlets, assign each to its own cluster 
        final_seqlet_clusters[seqlet_assigned==False] = np.array(
            range(len(pwm_cluster_remapping),
                  len(pwm_cluster_remapping)+sum(seqlet_assigned==False))) 

        final_seqlet_clusters = final_seqlet_clusters.astype("int")

        return final_seqlet_clusters
