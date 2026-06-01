% MinHash module tests adapted from lab 7

clear; clc;
set(0, 'DefaultFigureVisible', 'off');
addpath('../aux_functions');
addpath('../modules');

PATCH_SIZE = 64;
refPath    = '../data/authentic/The_Kiss_Real.jpg';
ref        = imread(refPath);
refGray    = rgb2gray(ref);

[H, W] = size(refGray);
nRow   = floor(H / PATCH_SIZE);
nCol   = floor(W / PATCH_SIZE);
nPatch = nRow * nCol;

% Trains MinHash with the reference image, extracting patches and computing their hashes
fprintf('Training MinHash (k=100, patch=%d, %d patches)\n', PATCH_SIZE, nPatch);
mhParams = minHash('train', refGray, PATCH_SIZE, 100);

passed = 0;  
failed = 0;

%% ══════════════════════════════════════════════════════════════════════
% TEST 1 – Self-similarity  (PL_07 7.1)
% Compares image with itself. A set compared against itself has Jaccard distance = 0, similarity = 1.0. 
% Base case: authentic painting vs. itself.
%% ══════════════════════════════════════════════════════════════════════
fprintf('\n══ TEST 1: Self-similarity ══\n');

res = minHash('test', refGray, mhParams);
fprintf('  The_Kiss_Real vs. itself  →  similarity = %.4f  (expected ≈ 1.0)\n', res.similarity);

if res.similarity > 0.9
    fprintf('  PASSED\n');
    passed = passed + 1;
else
    fprintf('  FAILED  (%.4f < 0.9)\n', res.similarity);
    failed = failed + 1;
end

%% ══════════════════════════════════════════════════════════════════════
% TEST 2 – Progressive patch corruption  (PL_07 7.1)
% Replaces x% of patches with random noise
% Compares the similarity obtained with the theoretical value (1-x)/(1+x)
%% ══════════════════════════════════════════════════════════════════════
fprintf('\n══ TEST 2: Progressive patch corruption ══\n');

pct_list = [0, 10, 20, 40, 60, 80, 100];
sims     = zeros(1, numel(pct_list));

fprintf('  %-22s  %-22s  %s\n', 'Corrupted patches (%)', 'obtained similarity', 'theoretical similarity');
fprintf('  %s\n', repmat('-', 1, 50));

for pi = 1:numel(pct_list)
    pct       = pct_list(pi);
    corrupted = refGray;

    if pct > 0
        nCorrupt  = round(nPatch * pct / 100);
        patchIdx  = randperm(nPatch, nCorrupt);
        for i = 1:nCorrupt
            pr = floor((patchIdx(i)-1) / nCol);
            pc = mod(patchIdx(i)-1,    nCol);
            r0 = pr * PATCH_SIZE + 1;
            c0 = pc * PATCH_SIZE + 1;
            corrupted(r0:r0+PATCH_SIZE-1, c0:c0+PATCH_SIZE-1) = ...
                randi([0,255], PATCH_SIZE, PATCH_SIZE, 'uint8');
        end
    end

    r        = minHash('test', corrupted, mhParams);
    sims(pi) = r.similarity;
    teorica  = (1 - pct/100) / (1 + pct/100);
    fprintf('  %-22.0f  %-12.4f  %.4f\n', pct, sims(pi), teorica);
end

if sims(1) > sims(end-1)
    fprintf('  PASSED  (similarity decreases: %.4f → %.4f)\n', sims(1), sims(end-1));
    passed = passed + 1;
else
    fprintf('  FAILED\n');
    failed = failed + 1;
end

%% ══════════════════════════════════════════════════════════════════════
% TEST 3 – Variation of k  (PL_07 7.1.4 and 7.1.5)
% Uses classic MinHash with k hash functions. Estimates Jaccard similarity as fraction of i where sig_i(A) == sig_i(B).
%% ══════════════════════════════════════════════════════════════════════
fprintf('\n══ TEST 3: Variation of k – Classic MinHash ══\n');

p_mh = 2147483647;  % 2^31-1, Mersenne prime for MinHash

% Set A: hashes of reference patches
patchesRef = extract_patches(refGray, PATCH_SIZE);
hashesA    = cellfun(@patch_hash, patchesRef);   % N×1
NA         = numel(hashesA);

% Set B: 70% patches identical to A + 30% replaced with noise
nCorruptB = round(NA * 0.30);
corrB     = refGray;
pIdx      = randperm(NA, nCorruptB);
for i = 1:nCorruptB
    pr = floor((pIdx(i)-1) / nCol);
    pc = mod(pIdx(i)-1,    nCol);
    r0 = pr*PATCH_SIZE+1;  c0 = pc*PATCH_SIZE+1;
    corrB(r0:r0+PATCH_SIZE-1, c0:c0+PATCH_SIZE-1) = randi([0,255], PATCH_SIZE, PATCH_SIZE, 'uint8');
end
patchesTest = extract_patches(corrB, PATCH_SIZE);
hashesB     = cellfun(@patch_hash, patchesTest);  % N×1

% Exact Jaccard via set intersection
inter_AB   = numel(intersect(hashesA, hashesB));
union_AB   = numel(union(hashesA,     hashesB));
jExact     = inter_AB / union_AB;

fprintf('  |A| = %d patches  |B| = %d patches\n', NA, numel(hashesB));
fprintf('  Exact intersection = %d  |  Exact union = %d\n', inter_AB, union_AB);
fprintf('  Exact Jaccard      = %.4f  (theoretical ≈ 0.5385)\n\n', jExact);

% for each hash function, compute the minimum hash value over all patches
% if the minimums match for both images, it counts as a Jaccard estimate
% larger k means more stable estimates (lower variance)
kValues = [25, 50, 100];
nTrials = 100;   % independent runs with different hash functions

fprintf('  %-5s  %-10s  %-10s  %-12s  %-12s\n', 'k', 'mean', 'std obtained', 'std theoretical', 'abs error');
fprintf('  %s\n', repmat('-', 1, 55));

stdByK = zeros(1, numel(kValues));
for ki = 1:numel(kValues)
    k         = kValues(ki);
    estimates = zeros(1, nTrials);

    for t = 1:nTrials
        a = randi([1, p_mh-1], k, 1);
        b = randi([0, p_mh-1], k, 1);

        % k×NA and k×NB — each row is a hash function, each column is an element
        allHA = mod(a * hashesA' + repmat(b, 1, NA), p_mh);   % k×NA
        allHB = mod(a * hashesB' + repmat(b, 1, NA), p_mh);   % k×NA

        sigA = min(allHA, [], 2);   % k×1: min of each function over A
        sigB = min(allHB, [], 2);   % k×1: min of each function over B

        estimates(t) = mean(sigA == sigB);
    end

    stdObtido  = std(estimates);
    stdTeorico = sqrt(jExact * (1 - jExact) / k);
    erroAbs    = abs(mean(estimates) - jExact);
    stdByK(ki) = stdObtido;

    fprintf('  %-5d  %-10.4f  %-10.4f  %-12.4f  %-12.4f\n', ...
            k, mean(estimates), stdObtido, stdTeorico, erroAbs);
end

% Criterion: std with k=100 < std with k=25 (more stable estimates)
if stdByK(end) < stdByK(1)
    fprintf('  PASSED  (std decreases with k: %.4f → %.4f)\n', stdByK(1), stdByK(end));
    passed = passed + 1;
else
    fprintf('  FAILED  (std did not decrease with k)\n');
    failed = failed + 1;
end

%% ══════════════════════════════════════════════════════════════════════
%  TEST 4 – Dataset images  (PL_07 7.2)
% Tests the similarity of the reference painting with a set of candidate images, including a known fake and distinct paintings. 
% The fake should have low similarity, while distinct paintings may have intermediate values.
%% ══════════════════════════════════════════════════════════════════════
fprintf('\n══ TEST 4: Dataset images  ══\n');

candidates = {
    '../data/test/The_Kiss_Real_Tests/The_Kiss_Real.jpg',         'The_Kiss_Real (test)',    'real_test';
    '../data/test/The_Kiss_Real_Tests/The_Kiss_Fake1.jpg',        'The_Kiss_Fake1',           'fake';
    '../data/authentic/Starry_Night.jpg',                    'Starry_Night (authentic)', 'distinct';
    '../data/authentic/The_Girl_With_The_Pearl_Earring.jpg', 'Girl_Pearl_Earring',       'distinct';
};

fprintf('  %-38s  %-8s  %s\n', 'image', 'similarity', 'type');
fprintf('  %s\n', repmat('-', 1, 58));

simFake = NaN;
simVals = zeros(1, size(candidates, 1));
for i = 1:size(candidates, 1)
    img        = rgb2gray(imread(candidates{i,1}));
    r          = minHash('test', img, mhParams);
    tipo       = candidates{i,3};
    simVals(i) = r.similarity;
    fprintf('  %-38s  %-8.4f  %s\n', candidates{i,2}, r.similarity, tipo);
    if strcmp(tipo, 'fake'), simFake = r.similarity; end
end

fprintf('\n  Reference scale (Test 2):\n');
fprintf('    0%% corrupted patches  → sim = %.4f\n', sims(1));
fprintf('   40%% corrupted patches  → sim = %.4f\n', sims(4));
fprintf('  100%% corrupted patches  → sim = %.4f\n', sims(end));

if ~isnan(simFake) && simFake < 0.3
    fprintf('  PASSED  (The_Kiss_Fake1: sim = %.4f < 0.3)\n', simFake);
    passed = passed + 1;
else
    fprintf('  FAILED  (fake image similarity unexpectedly high: %.4f)\n', simFake);
    failed = failed + 1;
end

%% ══════════════════════════════════════════════════════════════════════
fprintf('\n══════════════════════════════════════════════════════\n');
fprintf('  Final result: %d/%d tests passed\n', passed, passed + failed);
fprintf('══════════════════════════════════════════════════════\n');

set(0, 'DefaultFigureVisible', 'on');

teorica_list = (1 - pct_list/100) ./ (1 + pct_list/100);

% Bar chart – TEST 4: similarity per candidate image
figure('Visible', 'on', 'Color', 'white');
typeColors = containers.Map({'real_test','fake','distinct'}, ...
    {[0.20 0.55 0.85], [0.85 0.25 0.25], [0.50 0.75 0.40]});
nCand = size(candidates, 1);
barHandles = gobjects(nCand, 1);
hold on;
for i = 1:nCand
    barHandles(i) = bar(i, simVals(i), 0.55, 'FaceColor', typeColors(candidates{i,3}), 'EdgeColor', 'none');
end
yline(0.9, '--', 'Limiar real (0.9)', 'Color', [0.2 0.2 0.2], 'LineWidth', 1.2, 'LabelHorizontalAlignment', 'left');
hold off;
xticks(1:nCand);
xticklabels(candidates(:,2));
xtickangle(20);
ylabel('MinHash Similarity');
title('MinHash – Similaridade por imagem (Test 4)');
ylim([0, 1.1]);
legend([barHandles(find(strcmp(candidates(:,3),'real_test'), 1)), ...
        barHandles(find(strcmp(candidates(:,3),'fake'),      1)), ...
        barHandles(find(strcmp(candidates(:,3),'distinct'),  1))], ...
       {'Real', 'Fake', 'Distinto'}, 'Location', 'northeast');
grid on; grid minor;

figure;
plot(pct_list, sims,         'o-', 'LineWidth', 2, 'MarkerSize', 6, 'MarkerFaceColor', 'auto', 'DisplayName', 'obtida');
hold on;
plot(pct_list, teorica_list, 's--', 'LineWidth', 1.5, 'MarkerSize', 5, 'DisplayName', 'teórica');
hold off;
xlabel('Patches corrompidos (%)');
ylabel('Similarity');
title('MinHash – Similarity vs. corrupção de patches (Test 2)');
legend('Location', 'northeast');
xlim([-2, 102]);
ylim([-0.05, 1.05]);
grid on;
