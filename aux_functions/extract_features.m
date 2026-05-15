function feats = extract_features(img, minhash_score, bloom_ratio)
%   EXTRACT_FEATURES  Compute a 9-element feature vector from a painting image.
%
%   INPUT
%     img           - H x W x 3 uint8 RGB image 
%     minhash_score - scalar in [0,1] 
%     bloom_ratio   - scalar in [0,1]
%
%   OUTPUT
%     feats - (1 x 9) double row vector:
%               [r_mean, g_mean, b_mean,
%                glcm_contrast, glcm_energy, glcm_correlation,
%                edge_density,
%                minhash_score, bloom_ratio]

    r_mean = double(mean(img(:,:,1), 'all')) / 255;
    g_mean = double(mean(img(:,:,2), 'all')) / 255;
    b_mean = double(mean(img(:,:,3), 'all')) / 255;

    gray = double(rgb2gray(img));          

    NUM_LEVELS = 32;
    gray_q = floor(gray / 256 * NUM_LEVELS) + 1;
    gray_q = min(gray_q, NUM_LEVELS);    

    [rows, cols] = size(gray_q);
    glcm = zeros(NUM_LEVELS, NUM_LEVELS);
    for c = 1:cols-1
        for r = 1:rows
            i = gray_q(r, c);
            j = gray_q(r, c+1);
            glcm(i, j) = glcm(i, j) + 1;
            glcm(j, i) = glcm(j, i) + 1;   % symmetric
        end
    end

    % Normalise
    glcm = glcm / (sum(glcm(:)) + eps);

    % Pre-compute index grids for the stats formulas
    [I, J]   = meshgrid(1:NUM_LEVELS, 1:NUM_LEVELS);
    I = I'; J = J';

    mu_i = sum(sum(I .* glcm));
    mu_j = sum(sum(J .* glcm));
    sig_i = sqrt(sum(sum(glcm .* (I - mu_i).^2)) + eps);
    sig_j = sqrt(sum(sum(glcm .* (J - mu_j).^2)) + eps);

    glcm_contrast    = sum(sum(glcm .* (I - J).^2));
    glcm_energy      = sum(sum(glcm .^2));
    glcm_correlation = sum(sum(glcm .* (I - mu_i) .* (J - mu_j))) / (sig_i * sig_j);

    Kx = [-1 0 1; -2 0 2; -1 0 1];
    Ky = [-1 -2 -1;  0  0  0;  1  2  1];
    Gx = conv2(gray, Kx, 'same');
    Gy = conv2(gray, Ky, 'same');
    G  = sqrt(Gx.^2 + Gy.^2);
    threshold    = 0.1 * max(G(:));     
    edges        = G > threshold;
    edge_density = sum(edges(:)) / numel(edges);

    feats = [r_mean,        g_mean,       b_mean, ...
             glcm_contrast, glcm_energy,  glcm_correlation, ...
             edge_density, ...
             minhash_score, bloom_ratio];
end