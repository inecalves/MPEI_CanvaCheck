function result = bloomFilter(varargin)
%   BLOOMFILTER  Bloom Filter module for fake-painting detection.
%
%   --- TRAIN MODE ---
%   bfParams = bloomFilter('train', img, patchSize, bf_size, num_hashes)
%
%   --- TEST MODE ---
%   result = bloomFilter('test', testImg, bfParams)
%
%   OUTPUT (test mode)
%     result.membership_ratio - fraction of test patches found in the filter [0,1]

    if nargin < 1
        error('bloomFilter: specify mode as ''train'' or ''test''.');
    end
    mode = lower(varargin{1});

    switch mode
        case 'train'
            if nargin ~= 5
                error('bloomFilter train: usage  bfParams = bloomFilter(''train'', img, patchSize, bf_size, num_hashes)');
            end

            img        = varargin{2};
            patchSize  = varargin{3};
            bf_size    = varargin{4};
            num_hashes = varargin{5};

            % break the reference image into fixed-size patches
            patches  = extract_patches(img, patchSize);
            nPatches = numel(patches);

            if nPatches == 0
                error('bloomFilter train: image too small for patchSize=%d.', patchSize);
            end

            % bit array: all bits start at 0 (nothing inserted yet)
            bf_array = false(1, bf_size);

            % Parameters for num_hashes independent hash functions h_i(x) = (a_i*x + b_i) mod p
            p = 2147483647;  
            a = randi([1, p-1], 1, num_hashes);  % random slopes (one per hash function)
            b = randi([0, p-1], 1, num_hashes);  % random intercepts

            % insert every patch: hash it and set the corresponding bit positions to 1
            h = waitbar(0, 'BloomFilter: inserting patches...');
            for i = 1:nPatches
                waitbar(i / nPatches, h, sprintf('BloomFilter training: patch %d / %d', i, nPatches));
                hVal     = patch_hash(patches{i});         % single integer fingerprint of the patch
                posicoes = mod(a .* hVal + b, p);          % apply all hash functions
                posicoes = mod(posicoes, bf_size) + 1;     % map into [1, bf_size]
                bf_array(posicoes) = true;                 % mark those bit positions
            end
            close(h);

            result.bf_array   = bf_array;
            result.a          = a;
            result.b          = b;
            result.p          = p;
            result.bf_size    = bf_size;
            result.num_hashes = num_hashes;
            result.patchSize  = patchSize;

        case 'test'
            if nargin ~= 3
                error('bloomFilter test: usage  result = bloomFilter(''test'', testImg, bfParams)');
            end

            testImg  = varargin{2};
            bfParams = varargin{3};

            % restore the filter and hash parameters saved during training
            bf_array   = bfParams.bf_array;
            a          = bfParams.a;
            b          = bfParams.b;
            p          = bfParams.p;
            bf_size    = bfParams.bf_size;
            patchSize  = bfParams.patchSize;

            % extract patches from the image under test using the same patch size
            testPatches  = extract_patches(testImg, patchSize);
            nTestPatches = numel(testPatches);

            if nTestPatches == 0
                warning('bloomFilter test: test image too small; returning ratio = 0.');
                result = struct('membership_ratio', 0);
                return;
            end

            % query each test patch: it is considered a "member" only if all its hash positions are set to 1 in the filter 
            member_count = 0;
            h = waitbar(0, 'BloomFilter: querying patches...');
            for i = 1:nTestPatches
                waitbar(i / nTestPatches, h, sprintf('BloomFilter testing: patch %d / %d', i, nTestPatches));
                hVal     = patch_hash(testPatches{i});
                posicoes = mod(a .* hVal + b, p);
                posicoes = mod(posicoes, bf_size) + 1;
                if all(bf_array(posicoes))   % all bits set → patch probably seen during training
                    member_count = member_count + 1;
                end
            end
            close(h);

            % membership_ratio close to 1 → test image looks like the reference
            result.membership_ratio = member_count / nTestPatches;

        otherwise
            error('bloomFilter: unknown mode "%s". Use ''train'' or ''test''.', mode);
    end
end
