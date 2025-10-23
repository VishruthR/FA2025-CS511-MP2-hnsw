### TODOs
- [ ] Push to private repo and take screenshot of workflow working for part 0
- [ ] DiskANN
- [ ] Write up rest of answers
- [ ] Clean up graphs (have annotations appear on opposite side of point alternating fashion)

### Notes

used Docker container to run microsoft's diskann


## HNSW vs. LSH

efSearch decreases QPS dramatically while offering modest gains in recall. In particular, adjusting efSearch from 10 to 200 leads to about a 30x decrease in QPS but a 0.15 increase in recall. 

nbits offers a simlar tradeoff. Adjusting nbits from 32 to 768 leads to a 4x decrease in QPS but a 0.4 increase in recall.

HNSW seems to be the overall superior model with significantly better QPS for almost all efsearch values and significantly better recall across all parameters tested.