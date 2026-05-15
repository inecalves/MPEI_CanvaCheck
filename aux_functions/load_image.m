function img = load_image(path)
%   LOAD_IMAGE  Load an image from disk and ensure it is RGB 
%
%   INPUT
%     path - file path to the image
%
%   OUTPUT
%     img  - H x W x 3 uint8 RGB image

    raw = imread(path);

    if size(raw, 3) == 3
        img = raw;                      
    else
        img = repmat(raw, 1, 1, 3);   
    end
end