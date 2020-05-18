#!/bin/sh

# downloading the embedding models and creating TargetPred lexicons (all four versions) for all 91 languages

python download_embeddings.py &&
python TargetPred_MTLgrouped.py &&
python TargetPred_MTLall.py &&
python TargetPred_STL.py &&
python TargetPred_ridge.py