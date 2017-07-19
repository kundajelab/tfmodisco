
workflow would be like:

datatrack1
datatrack2

trackSets = maketrackset(datatrack1, datatrack2)

seqlets = TrackSet.createSeqlets(CoordProducer(...).getCoords)

clusters = LouvainCluster(DistanceMatrixFromSeqlets(seqlets, ...).getDistanceMatrix()).getClusters()
