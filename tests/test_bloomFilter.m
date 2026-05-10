% test_bloomFilter.m
% Tests how the membership_ratio changes as we corrupt more and more pixels
% A ratio near 1.0 means the image looks like the original
% A ratio near 0.0 means it looks completely different
% Note: the ratio never reaches exactly 0 due to false positives (hash collisions)

addpath('../aux_functions');
addpath('../modules');

patchSize = 32;

% train the filter on the real painting
ref      = imread('../data/authentic/The_Kiss_Real.jpg');
bfParams = bloomFilter('train', ref, patchSize, 500000, 5);

% convert to grayscale 
gray = double(rgb2gray(ref));
[H, W] = size(gray);

% only count pixels that belong to a full patch (ignore possible borders)
nRow = floor(H / patchSize);
nCol = floor(W / patchSize);
nPatchPixels = nRow * nCol * patchSize^2;

percentages = [0, 0.01, 0.05, 0.1, 0.5, 1, 5, 10, 25, 50, 100];

fprintf('\n%-20s  %s\n', 'Pixels altered (%)', 'membership_ratio');
fprintf('%s\n', repmat('-', 1, 42));

for pct = percentages
    corrupted = gray;

    if pct > 0
        % randomly replace pct% of patch pixels with noise
        n   = round(nPatchPixels * pct / 100);
        idx = randperm(nPatchPixels, n);

        flat = corrupted(1:nRow*patchSize, 1:nCol*patchSize);
        flat = flat(:);
        flat(idx) = randi([0, 255], n, 1);
        corrupted(1:nRow*patchSize, 1:nCol*patchSize) = reshape(flat, nRow*patchSize, nCol*patchSize);
    end

    r = bloomFilter('test', uint8(corrupted), bfParams);
    fprintf('%-20.2f  %.4f\n', pct, r.membership_ratio);
end

fprintf('\n(ratio ~1.0 = identical to reference,  ~0.0 = completely different)\n');
