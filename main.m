addpath('aux_functions');
addpath('modules');

ref        = imread('./data/authentic/The_Scream.jpg');
hashParams = minHash('train', ref, 32, 100);

test       = imread('./data/test/The_Scream_Tests/The_Scream_Fake9.jpg');
r          = minHash('test', test, hashParams);
fprintf('Similarity: %.4f  Distance: %.4f\n', r.similarity, r.distance);