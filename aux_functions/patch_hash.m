function h = patch_hash(patch)
%   PATCH_HASH  Hash a patch feature vector to a single non-negative integer.
%
%   INPUT
%     patch - column vector of pixel values (double, range 0-255)
%
%   OUTPUT
%     h     - hash

    p = uint8(patch);
    h = uint64(5381);
    for i = 1:numel(p)
        h = mod(h * uint64(33) + uint64(p(i)), uint64(2^32 - 1));
    end
    h = double(h);
end