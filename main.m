clear; clc; close all;
addpath('aux_functions');
addpath('modules');
addpath('data');

PATCH_SIZE  = 32;
MH_K        = 100;      % MinHash: number of hash functions
BF_SIZE     = 500000;   % Bloom Filter: bit-array length
BF_HASHES   = 5;        % Bloom Filter: number of hash functions
REAL_THRESH = 0.5;      % Naïve Bayes threshold: prob_real above this → classified as REAL

authDir   = './data/authentic/';

fprintf('══════════════════════════════════════════════════════\n');
fprintf('                IMAGE VERIFICATION DEMO               \n');
fprintf('══════════════════════════════════════════════════════\n');
fprintf('1. Custom test: Manually pick 1 Reference and 1 Test image\n');
fprintf('2. Folder test: Test a specific reference painting group\n');
fprintf('3. All test: Run the entire dataset automatically\n');
fprintf('══════════════════════════════════════════════════════\n');

userChoice = input('Select an option (1-3): ');

switch userChoice
    case 1
        % ==== Option 1: Manual Pick ====
        [refName, refPathDir] = uigetfile({'*.jpg;*.png;*.png'}, 'Select the Authentic REFERENCE Painting');
        if isequal(refName, 0), error('Operation cancelled by user.'); end
        refPath = fullfile(refPathDir, refName);
        
        [tName, tPathDir] = uigetfile({'*.jpg;*.png;*.png'}, 'Select the TEST Image');
        if isequal(tName, 0), error('Operation cancelled by user.'); end
        customTestPath = fullfile(tPathDir, tName);
        
        [~, refStem, ~] = fileparts(refName);
        authFiles = struct('name', refName, 'folder', refPathDir);
        
    case 2
        % ==== Option 2: Choose specific group from folder ====
        allAuth = dir(fullfile(authDir, '*.jpg'));
        if isempty(allAuth), error('No authentic images found in %s', authDir); end
        
        fprintf('\nAvailable reference paintings:\n');
        for i = 1:numel(allAuth)
            fprintf('  [%d] %s\n', i, allAuth(i).name);
        end
        idx = input(sprintf('Select a painting index (1-%d): ', numel(allAuth)));
        if idx < 1 || idx > numel(allAuth), error('Invalid index selected.'); end
        
        authFiles = allAuth(idx);
        
    case 3
        % ==== Option 3: Do all the groups that are in data ====
        authFiles = dir(fullfile(authDir, '*.jpg'));
        if isempty(authFiles), error('No authentic images found in %s', authDir); end
        
    otherwise
        error('Invalid selection. Exiting.');
end

fprintf('\nFound %d reference painting(s) to process.\n\n', numel(authFiles));

function label = label_from_filename(fname)
    if contains(lower(fname), 'fake')
        label = 0;
    else
        label = 1;
    end
end

for p = 1:numel(authFiles)
    if userChoice ~= 1
        refPath = fullfile(authFiles(p).folder, authFiles(p).name);
        [~, refStem, ~] = fileparts(authFiles(p).name);
    end

    fprintf('══════════════════════════════════════════════════════\n');
    fprintf('  Painting: %s\n', refStem);
    fprintf('══════════════════════════════════════════════════════\n');

    % ==== 1. Load reference image ====
    refImg     = imread(refPath);
    refImgGray = rgb2gray(refImg);

    % ==== 2. Train MinHash & Bloom Filter on the reference ====
    fprintf('[1/4] Training MinHash...       ');
    mhParams = minHash('train', refImgGray, PATCH_SIZE, MH_K);
    fprintf('done.\n');

    fprintf('[2/4] Training Bloom Filter...  ');
    bfParams = bloomFilter('train', refImgGray, PATCH_SIZE, BF_SIZE, BF_HASHES);
    fprintf('done.\n');

    % ==== 3. Build NB training set ====
    % Uses the authentic image itself + everything in the corresponding test folder
    testFolder = fullfile('./data/test/', [refStem '_Tests']);
    trainPaths  = {refPath};          
    trainLabels = [1];

    if userChoice == 1
        testFiles = struct('name', refName, 'folder', refPathDir); 
        testFiles(1).name = refName; 
    else
        if isfolder(testFolder)
            testFiles = dir(fullfile(testFolder, '*.jpg'));
            for f = 1:numel(testFiles)
                trainPaths{end+1}  = fullfile(testFiles(f).folder, testFiles(f).name);
                trainLabels(end+1) = label_from_filename(testFiles(f).name);
            end
        else
            warning('Test folder not found: %s\nSkipping Naïve Bayes training.', testFolder);
            trainPaths  = {};
            trainLabels = [];
        end
    end

    fprintf('[3/4] Training Naïve Bayes on %d images (%d real / %d fake)...\n', ...
            numel(trainPaths), sum(trainLabels==1), sum(trainLabels==0));

    if numel(unique(trainLabels)) < 2
        if userChoice == 1
            fprintf('(Single mode: Skipping training, using basic weights if supported)\n');
        else
            warning('Need at least one real AND one fake image to train NB. Skipping...');
        end
        nbModel = [];
    else
        fprintf('[3/4] Training Naïve Bayes on %d images (%d real / %d fake)...\n', ...
                numel(trainPaths), sum(trainLabels==1), sum(trainLabels==0));
        nbModel = naiveBayes('train', trainPaths, trainLabels, mhParams, bfParams);
        fprintf('done.\n');
    end
    fprintf('done.\n');

    % ==== 4. Preparation for Testing loop ====
    if userChoice == 1
        nTests = 1;
    else
        if ~isfolder(testFolder)
            fprintf('(no test folder — nothing to test)\n\n');
            continue;
        end
        nTests = numel(testFiles);
    end

    fprintf('[4/4] Testing images...\n\n');

    % ==== Summary table storage ====
    imgNames   = cell(nTests, 1);
    mhScores   = zeros(nTests, 1);
    bfRatios   = zeros(nTests, 1);
    probReals  = zeros(nTests, 1);
    predLabels = zeros(nTests, 1);
    trueLabels = zeros(nTests, 1);

    fig = [];

    for f = 1:nTests
        if userChoice == 1
            tPath  = customTestPath;
            [~, tFileName, tExt] = fileparts(customTestPath);
            tFilesName = [tFileName tExt];
            tLabel = NaN;
        else
            tPath  = fullfile(testFiles(f).folder, testFiles(f).name);
            tFilesName = testFiles(f).name;
            tLabel = label_from_filename(tFilesName);
        end
        
        tImg   = imread(tPath);
        tGray  = rgb2gray(tImg);

        % ==== Run MinHash ====
        mhResult = minHash('test', tGray, mhParams);

        % ==== Run Bloom Filter ====
        bfResult = bloomFilter('test', tGray, bfParams);

        % ==== Run Naïve Bayes (if trained) ====
        if ~isempty(nbModel)
            nbResult  = naiveBayes('test', tImg, nbModel);
            prob_real = nbResult.prob_real;
            pred      = nbResult.label;
        else
            prob_real = (mhResult.similarity + bfResult.membership_ratio) / 2;
            if prob_real >= 0.7
                pred = 1;
            else
                pred = 0;
            end
        end

        imgNames{f}   = tFilesName;
        mhScores(f)   = mhResult.similarity;
        bfRatios(f)   = bfResult.membership_ratio;
        probReals(f)  = prob_real;
        predLabels(f) = pred;
        trueLabels(f) = tLabel;

        % ==== Console status updates ====
        if pred == 1, predStr = 'REAL'; elseif pred == 0, predStr = 'FAKE'; else, predStr = '  ? '; end

        if userChoice == 1
            fprintf('  %-40s  MH=%.3f  BF=%.3f  prob_real=%.3f  pred=%-4s\n', ...
                    tFilesName, mhResult.similarity, bfResult.membership_ratio, prob_real, predStr);
        else
            if tLabel == 1, trueStr = 'REAL'; else, trueStr = 'FAKE'; end
            fprintf('  %-40s  MH=%.3f  BF=%.3f  prob_real=%.3f  pred=%-4s  true=%-4s\n', ...
                    tFilesName, mhResult.similarity, bfResult.membership_ratio, prob_real, predStr, trueStr);
        end

        if isempty(fig) || ~ishandle(fig)
            fig = figure('Name', '', 'NumberTitle', 'off', 'Position', [100 100 1000 420]);
        end

        set(fig, 'Name', sprintf('%s — %s', refStem, tFilesName));

        subplot(1,2,1);
        imshow(refImg);
        title(sprintf('REFERENCE\n%s', refStem), 'Interpreter','none');

        subplot(1,2,2);
        imshow(tImg);

        if userChoice == 1
            title(sprintf('TEST IMAGE\nMinHash=%.3f   BloomFilter=%.3f\nPrediction → %s  (prob_real=%.3f)', ...
                  mhResult.similarity, bfResult.membership_ratio, predStr, prob_real), ...
                  'Interpreter', 'none');
        else
            if tLabel == 1, trueStr = 'REAL'; else, trueStr = 'FAKE'; end
            title(sprintf('TEST IMAGE  [true: %s]\nMinHash=%.3f   BloomFilter=%.3f\nPrediction → %s  (prob\\_real=%.3f)', ...
                  trueStr, mhResult.similarity, bfResult.membership_ratio, predStr, prob_real), ...
                  'Interpreter', 'tex');
        end

        drawnow;
    end

    % ==== 5. Summary table for this painting ====
    fprintf('\n── Summary: %s ──\n', refStem);
    if userChoice == 1
        fprintf('  %-40s  %6s  %6s  %9s  %4s\n', 'File', 'MinHash', 'Bloom', 'prob_real', 'Pred');
        fprintf('  %s\n', repmat('-', 1, 72));
        for f = 1:nTests
            if predLabels(f)==1, ps='REAL'; elseif predLabels(f)==0, ps='FAKE'; else, ps='?'; end
            fprintf('  %-40s  %6.3f  %6.3f  %9.3f  %4s\n', ...
                    imgNames{f}, mhScores(f), bfRatios(f), probReals(f), ps);
        end
    else
        fprintf('  %-40s  %6s  %6s  %9s  %4s  %4s\n', 'File', 'MinHash', 'Bloom', 'prob_real', 'Pred', 'True');
        fprintf('  %s\n', repmat('-', 1, 78));
        for f = 1:nTests
            if predLabels(f)==1, ps='REAL'; elseif predLabels(f)==0, ps='FAKE'; else, ps='?'; end
            if trueLabels(f)==1, ts='REAL'; else, ts='FAKE'; end
            fprintf('  %-40s  %6.3f  %6.3f  %9.3f  %4s  %4s\n', ...
                    imgNames{f}, mhScores(f), bfRatios(f), probReals(f), ps, ts);
        end
    end

    % ==== 6. Accuracy (only for option 2 and 3) ====
    if userChoice ~= 1
        known = ~isnan(predLabels) & ~isnan(trueLabels);
        if any(known) && ~isempty(nbModel)
            acc = mean(predLabels(known) == trueLabels(known)) * 100;
            fprintf('\n  Overall accuracy on test set: %.1f%%\n', acc);
        end
    end
    fprintf('\n');
end

fprintf('══════════════════════════════════════════════════════\n');
fprintf('  SCRIPT complete.\n');
fprintf('══════════════════════════════════════════════════════\n');