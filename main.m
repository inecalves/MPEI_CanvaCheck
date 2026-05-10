addpath('aux_functions');
addpath('modules');

ref        = imread('./data/authentic/The_Scream.jpg');
hashParams = minHash('train', ref, 32, 100);

test       = imread('./data/test/The_Scream_Tests/The_Scream_Fake9.jpg');
r          = minHash('test', test, hashParams);
fprintf('Similarity: %.4f  Distance: %.4f\n', r.similarity, r.distance);

% --- Bloom Filter ---
ref_kiss    = imread('./data/authentic/The_Kiss_Real.jpg');
bfParams    = bloomFilter('train', ref_kiss, 32, 500000, 5);

fake_kiss   = imread('./data/test/The_Kiss_Tests/The_Kiss_Fake1.jpg');

r_bf_real   = bloomFilter('test', ref_kiss,  bfParams);
r_bf_fake   = bloomFilter('test', fake_kiss, bfParams);

fprintf('Bloom Filter - The Kiss (real): %.4f\n', r_bf_real.membership_ratio);
fprintf('Bloom Filter - The Kiss (fake): %.4f\n', r_bf_fake.membership_ratio);