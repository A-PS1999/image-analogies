# Hertzmann algorithm

## Create image analogy function algorithm overview
1. Compute gaussian pyramids for `A`, `A'` and `B`.
2. Compute feature vectors for `A`, `A'` and `B`. 
3. Initialize search structures for neighbor searching.  
&emsp;This includes data structure `s` (source) which is indexed by `q` and returns the pixel `p` that is the source of each index `q` aka `s(q) = p`.
4. For each level &ell; of pyramid, from coarsest to finest:  
For each pixel `q` for the output `B'` in scan-line order:  
&emsp;Assign `p` with output of `BestMatch(A, A', B, B', s, l, q)`  
&emsp;Assign `A'(p)` (for current level &ell;) to `B'(q)` for that level
&emsp;Insert `p` into `s` at index `q`
5. Finally return synthesized `B'`

## BestMatch function overview
1. Find the closest-matching pixel p in the source using `BestApproximateMatch(A, A', B, B', l, q)` aka `p_app`
2. Find the closest-matching pixel using `BestCoherenceMatch(A, A', B, B', s, l, q)` aka `p_coh`
2. Calculate distance between feature vector for `p_app` at current level and feature vector for `q` at current level aka `||Fl(p_app) - Fl(q)||²`. This is `d_app`.
3. Do the same for `p_coh` aka `||Fl(p_coh) - Fl(q)||²`. This is `d_coh`.
4. If `d_coh` < `d_app`(1 + 2<sup>&ell;-L</sup>K) then return `d_coh` else `d_app`
* K is a coherence parameter used to weight whether accuracy or coherence is favoured in the output.
* F&ell;(`p`) denotes concatenation of all feature vectors within a neighbourhood `N(p)` of both source images `A` and `A'` at the level &ell; and &ell; - 1.

`BestApproximateMatch` uses ANN (Approximate Nearest Neighbour) search. Could use PatchMatch here.

`BestCoherenceMatch` returns `s(r*) + (q - r*)` where `r*` is defined as the minimum `r ∈ N(q)` for the following calculation:  
&emsp;`||Fl(s(r) + (q - r)) - Fl(q)||²`  
`N(q)` is the neighbourhood of already synthesized pixels adjacent to `q` in `B'` for that level.