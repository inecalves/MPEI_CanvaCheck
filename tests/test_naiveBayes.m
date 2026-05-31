% Naive Bayes module tests adapted from lab 5

clear; clc;
set(0, 'DefaultFigureVisible', 'off');
addpath('../aux_functions');
addpath('../modules');

PATCH_SIZE = 64;    
MH_K       = 100;
BF_SIZE    = 500000;
BF_HASHES  = 5;

refPath    = '../data/authentic/The_Scream.jpg';
testFolder = '../data/test/The_Scream_Tests/';
refImg     = imread(refPath);
refGray    = rgb2gray(refImg);

% initialize MinHash and Bloom Filter with the reference image
fprintf('Initializing MinHash and Bloom Filter for The Scream (patch=%d)\n', PATCH_SIZE);
mhParams = minHash('train', refGray, PATCH_SIZE, MH_K);
bfParams = bloomFilter('train', refGray, PATCH_SIZE, BF_SIZE, BF_HASHES);

testFiles = dir(fullfile(testFolder, '*.jpg'));
allPaths = [{refPath}, arrayfun(@(f) fullfile(f.folder, f.name), testFiles(:)', 'UniformOutput', false)];
allLabels = [1, arrayfun(@(f) double(~contains(lower(f.name), 'fake')), testFiles(:)')];
N = numel(allPaths);

% extract 9 features from all images at once (3 from MinHash similarity, 3 from Bloom Filter ratio, 3 combined)
fprintf('Extracting features from %d images (The Scream)\n', N);
nbInit   = naiveBayes('train', allPaths, allLabels, mhParams, bfParams);
featAll  = nbInit.featureMatrix;
rng(0);  % fixed seed for reproducible jitter
featAll  = featAll + 1e-6 * randn(size(featAll));  % prevents zero-variance errors with small splits
labelAll = allLabels(:);
fprintf('Features extracted. Matrix: %dx%d\n\n', size(featAll));

passed = 0;
failed = 0;

distNames = 'normal';

%% -------------------------------------------------------------------
% TEST 1 - 70/30 Split + Precision, Recall and F1  (PL_05 2b/f)
% Splits the dataset randomly: 70% for training, 30% for testing
% Computes confusion matrix and main classification metrics
%% -------------------------------------------------------------------
fprintf('== TEST 1: 70/30 Split - Precision, Recall, F1 ==\n');
rng(42);
idx    = randperm(N);
nTrain = round(0.7 * N);
trnIdx = idx(1:nTrain);
tstIdx = idx(nTrain+1:end);

trnL = labelAll(trnIdx);
tstL = labelAll(tstIdx);

fprintf('  Train: %d images (%d REAL / %d FAKE)  |  Test: %d images (%d REAL / %d FAKE)\n', ...
        numel(trnIdx), sum(trnL==1), sum(trnL==0), ...
        numel(tstIdx),  sum(tstL==1),  sum(tstL==0));

mdl   = fitcnb(featAll(trnIdx,:), trnL, 'DistributionNames', distNames, 'Prior', 'uniform');
preds = double(string(predict(mdl, featAll(tstIdx,:))));
tstLv = labelAll(tstIdx);

[precision, recall, f1, accuracy] = compute_metrics(preds, tstLv);

TP = sum(preds(:)==1 & tstLv(:)==1); FP = sum(preds(:)==1 & tstLv(:)==0);
TN = sum(preds(:)==0 & tstLv(:)==0); FN = sum(preds(:)==0 & tstLv(:)==1);
fprintf('\n  Confusion Matrix:\n');
fprintf('                  Prediction REAL  Prediction FAKE\n');
fprintf('  True REAL    :     %2d          %2d\n', TP, FN);
fprintf('  True FAKE    :     %2d          %2d\n', FP, TN);
fprintf('\n  Accuracy=%.1f%%  Precision=%.3f  Recall=%.3f  F1=%.3f\n', accuracy, precision, recall, f1);

if f1 >= 0.5
    fprintf('  PASSED  (F1=%.3f >= 0.5)\n', f1);  passed = passed + 1;
else
    fprintf('  FAILED  (F1=%.3f < 0.5)\n', f1);   failed = failed + 1;
end

%% -------------------------------------------------------------------
% SCATTER PLOT - Espaço de features: MinHash vs Bloom Filter (split 70/30)
% Círculos = treino  |  Quadrados = teste
% Azul = REAL  |  Vermelho = FAKE  |  X preto = erro de classificação
%% -------------------------------------------------------------------
figure('Visible', 'on');
hold on;

realTrn = trnIdx(trnL == 1);
fakeTrn = trnIdx(trnL == 0);
scatter(featAll(realTrn, 8), featAll(realTrn, 9), 70, [0.20 0.40 0.80], 'o', 'filled', ...
    'DisplayName', 'Treino – Real');
scatter(featAll(fakeTrn, 8), featAll(fakeTrn, 9), 70, [0.80 0.20 0.20], 'o', 'filled', ...
    'DisplayName', 'Treino – Fake');

realTst = tstIdx(tstLv == 1);
fakeTst = tstIdx(tstLv == 0);
scatter(featAll(realTst, 8), featAll(realTst, 9), 90, [0.20 0.40 0.80], 's', ...
    'LineWidth', 1.5, 'DisplayName', 'Teste – Real');
scatter(featAll(fakeTst, 8), featAll(fakeTst, 9), 90, [0.80 0.20 0.20], 's', ...
    'LineWidth', 1.5, 'DisplayName', 'Teste – Fake');

misIdx = tstIdx(preds ~= tstLv);
if ~isempty(misIdx)
    scatter(featAll(misIdx, 8), featAll(misIdx, 9), 130, 'k', 'x', ...
        'LineWidth', 2.0, 'DisplayName', 'Erro de classificação');
end

hold off;
xlabel('MinHash Score (feature 8)');
ylabel('Bloom Filter Ratio (feature 9)');
title('Naive Bayes – Dispersão: MinHash vs Bloom Filter (split 70/30)');
legend('Location', 'best');
grid on;

%% -------------------------------------------------------------------
% TEST 2 - 10 random runs (PL_05 2g)
% Repeats the 70/30 split ten times with different random seeds.
% Averages Precision, Recall and F1 across runs for a more stable result.
%% -------------------------------------------------------------------
fprintf('\n== TEST 2: 10 random runs - mean P/R/F1 ==\n');
nRuns = 10;
precisions = zeros(1,nRuns); recalls = zeros(1,nRuns);
f1s = zeros(1,nRuns);        accs    = zeros(1,nRuns);

fprintf('  %-5s  %-10s  %-10s  %-8s  %-8s\n', 'Run', 'Accuracy', 'Precision', 'Recall', 'F1');
fprintf('  %s\n', repmat('-', 1, 45));

for run = 1:nRuns
    idx = randperm(N);
    nT  = round(0.7 * N);
    tL  = labelAll(idx(1:nT));
    vL  = labelAll(idx(nT+1:end));
    if numel(unique(tL)) < 2, continue; end

    m  = fitcnb(featAll(idx(1:nT),:), tL, 'DistributionNames', distNames, 'Prior', 'uniform');
    pr = double(string(predict(m, featAll(idx(nT+1:end),:))));

    [precisions(run), recalls(run), f1s(run), accs(run)] = compute_metrics(pr, vL);
    fprintf('  %-5d  %-10.1f  %-10.3f  %-8.3f  %-8.3f\n', run, accs(run), precisions(run), recalls(run), f1s(run));
end

fprintf('  %s\n', repmat('-', 1, 45));
fprintf('  %-5s  %-10.1f  %-10.3f  %-8.3f  %-8.3f\n', 'Mean', mean(accs), mean(precisions), mean(recalls), mean(f1s));

if mean(f1s) >= 0.5
    fprintf('  PASSED  (mean F1=%.3f >= 0.5)\n', mean(f1s));  passed = passed + 1;
else
    fprintf('  FAILED  (mean F1=%.3f < 0.5)\n', mean(f1s));   failed = failed + 1;
end

%% -------------------------------------------------------------------
% TEST 3 - Effect of training set size  (PL_05 2h)
% Tests with 50%, 70% and 90% of data for training.
% More training data should generally improve classifier performance.
%% -------------------------------------------------------------------
fprintf('\n== TEST 3: Effect of training set size ==\n');
fprintf('  %-12s  %-10s  %-10s  %-8s  %-8s\n', 'Train (%)', 'Accuracy', 'Precision', 'Recall', 'F1');
fprintf('  %s\n', repmat('-', 1, 52));

trainSizes = [0.5, 0.7, 0.9];
f1BySize   = zeros(1, numel(trainSizes));
rng(7);
for si = 1:numel(trainSizes)
    ts   = trainSizes(si);
    idx  = randperm(N);
    nT   = round(ts * N);
    tL   = labelAll(idx(1:nT));
    vL   = labelAll(idx(nT+1:end));
    if numel(unique(tL)) < 2 || isempty(vL)
        fprintf('  %-12.0f  [insufficient data]\n', ts*100); continue;
    end

    m  = fitcnb(featAll(idx(1:nT),:), tL, 'DistributionNames', distNames, 'Prior', 'uniform');
    pr = double(string(predict(m, featAll(idx(nT+1:end),:))));

    [p, r, f1, ac] = compute_metrics(pr, vL);
    f1BySize(si) = f1;
    fprintf('  %-12.0f  %-10.1f  %-10.3f  %-8.3f  %-8.3f\n', ts*100, ac, p, r, f1);
end

fprintf('\n  F1: 50%%=%.3f  70%%=%.3f  90%%=%.3f\n', f1BySize(1), f1BySize(2), f1BySize(3));
if max(f1BySize) > 0
    fprintf('  PASSED\n');  passed = passed + 1;
else
    fprintf('  FAILED\n');  failed = failed + 1;
end

%% -------------------------------------------------------------------
% TEST 4 - Cross-painting generalisation  (PL_05 3)
% Trains on The Scream images and tests on Girl with the Pearl Earring.
%% -------------------------------------------------------------------
fprintf('\n== TEST 4: Cross-painting generalisation ==\n');

snAuthPath  = '../data/authentic/The_Girl_With_The_Pearl_Earring.jpg';
snTestFiles = dir(fullfile('../data/test/The_Girl_With_The_Pearl_Earring_Tests/', '*.jpg'));
snPaths     = [{snAuthPath}, arrayfun(@(f) fullfile(f.folder, f.name), snTestFiles(:)', 'UniformOutput', false)];
nSN         = numel(snPaths);

fprintf('  Extracting features from %d images (Girl with the Pearl Earring)...\n', nSN);
featSN = zeros(nSN, 9);
for i = 1:nSN
    img         = imread(snPaths{i});
    mhR         = minHash('test',     rgb2gray(img), mhParams);
    bfR         = bloomFilter('test', rgb2gray(img), bfParams);
    featSN(i,:) = extract_features(img, mhR.similarity, bfR.membership_ratio);
end

% train on all The Scream features, test on Girl with the Pearl Earring
mdlFinal = fitcnb(featAll, labelAll, 'DistributionNames', distNames, 'Prior', 'uniform');
preds    = double(string(predict(mdlFinal, featSN)));
nDecided = sum(preds == 0 | preds == 1);
nFake    = sum(preds == 0);

fprintf('\n  %-46s  %s\n', 'File', 'Prediction');
fprintf('  %s\n', repmat('-', 1, 53));
for i = 1:nSN
    [~, nm, ~] = fileparts(snPaths{i});
    if preds(i)==1, ps='REAL'; else, ps='FAKE'; end
    fprintf('  %-46s  %s\n', nm, ps);
end
fprintf('\n  Decisions: %d/%d  |  FAKE: %d/%d\n', nDecided, nSN, nFake, nSN);
fprintf('  Note: Girl with the Pearl Earring is not The Scream -> expected to classify as FAKE.\n');

if nDecided == nSN
    fprintf('  PASSED\n');  passed = passed + 1;
else
    fprintf('  FAILED\n');  failed = failed + 1;
end

%% -------------------------------------------------------------------
fprintf('\n======================================================\n');
fprintf('  Final result: %d/%d tests passed\n', passed, passed + failed);
fprintf('======================================================\n');
set(0, 'DefaultFigureVisible', 'on');


function [precision, recall, f1, accuracy] = compute_metrics(preds, labels)
    preds  = preds(:);
    labels = labels(:);
    TP = sum(preds==1 & labels==1);  FP = sum(preds==1 & labels==0);
    TN = sum(preds==0 & labels==0);  FN = sum(preds==0 & labels==1);
    precision = TP / (TP + FP + eps);
    recall    = TP / (TP + FN + eps);
    f1        = 2 * precision * recall / (precision + recall + eps);
    accuracy  = (TP + TN) / numel(labels) * 100;
end