function result = naiveBayes(varargin)
%   NAIVEBAYES  Naïve Bayes classifier module for fake-painting detection.
%
%   --- TRAIN MODE ---
%   model = naiveBayes('train', imgPaths, labels, minHashParams, bfParams)
%
%   --- TEST MODE ---
%   result = naiveBayes('test', testImg, model)
%
%   OUTPUT (test mode)
%     result.label       - 0 (fake) or 1 (real)
%     result.prob_real   - posterior probability of being real [0,1]
%     result.prob_fake   - posterior probability of being fake [0,1]
%     result.features    - feature vector used for classification

    if nargin < 1
        error('naiveBayes: specify mode as ''train'' or ''test''.');
    end
    mode = lower(varargin{1});

    switch mode
        case 'train'
            if nargin ~= 5
                error('naiveBayes train: usage  model = naiveBayes(''train'', imgPaths, labels, minHashParams, bfParams)');
            end

            imgPaths      = varargin{2};   % cell array of paths
            labels        = varargin{3};   % 0/1 vector
            minHashParams = varargin{4};
            bloomFilterParams= varargin{5};

            nImages = numel(imgPaths);
            if nImages ~= numel(labels)
                error('naiveBayes train: imgPaths and labels must have the same length.');
            end

            fprintf('naiveBayes: extracting features from %d training images...\n', nImages);

            featureMatrix = zeros(nImages, 9);   % 9 features per image 

            for i = 1:nImages
                img = load_image(imgPaths{i});
                mhResult = minHash('test', img, minHashParams);
                bfResult = bloomFilter('test', img, bloomFilterParams);
                featureMatrix(i, :) = extract_features(img, mhResult.similarity, bfResult.membership_ratio);
                fprintf('  [%d/%d] %s  label=%d\n', i, nImages, imgPaths{i}, labels(i));
            end

            nbModel = fitcnb(featureMatrix, labels(:), 'DistributionNames', 'kernel');

            result.nbModel          = nbModel;
            result.minHashParams    = minHashParams;
            result.bloomFilterParams= bloomFilterParams;
            result.featureMatrix    = featureMatrix;
            result.labels           = labels(:);

        case 'test'
            if nargin ~= 3
                error('naiveBayes test: usage  result = naiveBayes(''test'', testImg, model)');
            end

            testImg = varargin{2};
            model   = varargin{3};

            mhResult = minHash('test',     testImg, model.minHashParams);
            bfResult = bloomFilter('test', testImg, model.bloomFilterParams);

            feats = extract_features(testImg, mhResult.similarity, bfResult.membership_ratio);

            % Naïve Bayes prediction
            [label, posterior] = predict(model.nbModel, feats);

            result.label      = label;
            result.prob_real  = posterior(2);   
            result.prob_fake  = posterior(1);   
            result.features   = feats;
            result.minhash_similarity  = mhResult.similarity;
            result.bloom_membership    = bfResult.membership_ratio;

        otherwise
            error('naiveBayes: unknown mode "%s". Use ''train'' or ''test''.', mode);
    end
end