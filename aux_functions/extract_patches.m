function patches = extract_patches(img, patchSize)
%   EXTRACT_PATCHES  Divide a grayscale image into non-overlapping square patches.
%
%   INPUT
%     img       - 2-D grayscale image (uint8 or double)
%     patchSize - side length in pixels (e.g. 32)
%
%   OUTPUT
%     patches   - (nPatches x 1) cell array; each cell is a (patchSize^2 x 1) double column vector of pixel values in [0,255]

    if size(img, 3) > 1
        img = rgb2gray(img);
    end
    img = double(img);

    [rows, cols] = size(img);
    nRow = floor(rows / patchSize);
    nCol = floor(cols / patchSize);

    patches = cell(nRow * nCol, 1);
    k = 1;
    for r = 1:nRow
        for c = 1:nCol
            r0 = (r-1)*patchSize + 1;  r1 = r*patchSize;
            c0 = (c-1)*patchSize + 1;  c1 = c*patchSize;
            patches{k} = img(r0:r1, c0:c1);
            patches{k} = patches{k}(:);   % flatten to column vector
            k = k + 1;
        end
    end
end