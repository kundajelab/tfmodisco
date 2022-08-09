For PhenoGraph, the C++ louvain code has been adapted to increase speed.
Specifically, graphs are written directly to binary format and the rewritten
louvain binary "convert" is expecting a binary file to operate on.