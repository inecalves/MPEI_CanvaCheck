function [similarity, distance] = jaccard_from_signatures(sig1, sig2)
%   JACCARD_FROM_SIGNATURES  Estimate Jaccard similarity from two MinHash signatures.
%   [similarity, distance] = jaccard_from_signatures(sig1, sig2)
%
%   INPUT
%     sig1, sig2 - (k x 1) signature vectors produced by minHash() - need to have been built with the SAME hashParams
%   OUTPUT
%     similarity - estimated Jaccard similarity in [0, 1] - (1 = identical sets, 0 = completely different)
%     distance   - 1 - similarity  (Jaccard distance)

    if numel(sig1) ~= numel(sig2)
        error('jaccard_from_signatures: signatures must have the same length k.');
    end

    k = numel(sig1);
    similarity = sum(sig1 == sig2) / k;
    distance   = 1 - similarity;
end