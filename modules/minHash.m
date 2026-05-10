function result = minHash(varargin)
%   MINHASH  MinHash module for fake-painting detection.
%
%   --- TRAIN MODE ---
%   hashParams = minHash('train', img, patchSize, k)
%
%   --- TEST MODE ---
%   result = minHash('test', testImg, hashParams)
%
%   OUTPUT (test mode)
%     result.similarity - scalar in [0,1]
%     result.distance   - 1 - similarity

    if nargin < 1
        error('minHash: specify mode as ''train'' or ''test''.');
    end
    mode = lower(varargin{1});

    switch mode
        case 'train'
            if nargin ~= 4
                error('minHash train: usage  hashParams = minHash(''train'', img, patchSize, k)');
            end

            img       = varargin{2};
            patchSize = varargin{3};
            k         = varargin{4};

            patches  = extract_patches(img, patchSize);
            nPatches = numel(patches);

            if nPatches == 0
                error('minHash train: image too small for patchSize=%d.', patchSize);
            end

            p = 2147483647;             % 2^31 - 1, Mersenne prime
            a = randi([1, p-1], 1, k);
            b = randi([0, p-1], 1, k);

            % signatures(:, i) = MinHash signature of patch i - each column is one patch's signature (k x 1)
            signatures = zeros(k, nPatches);
            h = waitbar(0, 'MinHash: computing reference signatures...');
            for i = 1:nPatches
                waitbar(i / nPatches, h, sprintf('MinHash training: patch %d / %d', i, nPatches));
                hVal = patch_hash(patches{i});
                signatures(:, i) = mod(a .* hVal + b, p)';
            end
            close(h);

            result.signatures = signatures;  % k x nPatches
            result.a          = a;
            result.b          = b;
            result.p          = p;
            result.patchSize  = patchSize;
            result.k          = k;

        case 'test'
            if nargin ~= 3
                error('minHash test: usage  result = minHash(''test'', img, hashParams)');
            end

            testImg    = varargin{2};
            hashParams = varargin{3};

            required = {'signatures','a','b','p','patchSize','k'};
            for f = required
                if ~isfield(hashParams, f{1})
                    error('minHash test: hashParams is missing field "%s".', f{1});
                end
            end

            k         = hashParams.k;
            patchSize = hashParams.patchSize;
            a         = hashParams.a;
            b         = hashParams.b;
            p         = hashParams.p;
            refSigs   = hashParams.signatures;  % k x nRefPatches

            testPatches  = extract_patches(testImg, patchSize);
            nTestPatches = numel(testPatches);

            if nTestPatches == 0
                warning('minHash test: test image too small; returning distance = 1.');
                result = struct('similarity', 0, 'distance', 1);
                return;
            end

            % ==== Compute signature for each test patch (k x nTestPatches) ====
            testSigs = zeros(k, nTestPatches);
            h2 = waitbar(0, 'MinHash: computing test signatures...');
            for i = 1:nTestPatches
                waitbar(i / nTestPatches, h2, sprintf('MinHash testing: patch %d / %d', i, nTestPatches));
                hVal = patch_hash(testPatches{i});
                testSigs(:, i) = mod(a .* hVal + b, p)';
            end
            close(h2);

            % ==== Patch matching ====
            % For each test patch, check if any reference patch has a nearly identical signature (Jaccard distance < threshold).
            % This estimates: |test_patches intersect ref_patches| / |test_patches union ref_patches|

            nRefPatches = size(refSigs, 2);
            matchThreshold = 0.05;  % signatures must agree on 95%+ of hash functions

            matched = 0;
            h3 = waitbar(0, 'MinHash: matching patches...');
            for i = 1:nTestPatches
                waitbar(i / nTestPatches, h3, sprintf('MinHash matching: patch %d / %d', i, nTestPatches));
                for j = 1:nRefPatches
                    [~, dist] = jaccard_from_signatures(testSigs(:,i), refSigs(:,j));
                    if dist < matchThreshold
                        matched = matched + 1;
                        break;  
                    end
                end
            end
            close(h3);
 
            % Jaccard over patch sets: intersection / union
            union_size = nTestPatches + nRefPatches - matched;
            similarity = matched / union_size;
            distance   = 1 - similarity;

            result = struct('similarity', similarity, 'distance', distance);

        otherwise
            error('minHash: unknown mode "%s". Use ''train'' or ''test''.', mode);
    end
end