% extractColorFeaturesTomato.m
function features = extractColorFeaturesTomato(img)
    if size(img,3) ~= 3 && size(img,3) ~= 1
        error('Input gambar harus RGB (3 channel) atau Grayscale (1 channel).');
    end
    if size(img,3) == 1 % Jika grayscale, duplikasi channel agar jadi RGB semu
        img = cat(3, img, img, img);
    end

    % Fitur RGB
    R = double(img(:,:,1));
    G = double(img(:,:,2));
    B = double(img(:,:,3));

    meanR = mean(R(:)); stdR = std(R(:));
    meanG = mean(G(:)); stdG = std(G(:));
    meanB = mean(B(:)); stdB = std(B(:));

    % Fitur HSV
    try
        imgHSV = rgb2hsv(img);
        H = double(imgHSV(:,:,1)); % Hue
        S = double(imgHSV(:,:,2)); % Saturation
        V = double(imgHSV(:,:,3)); % Value

        meanH = mean(H(:)); stdH = std(H(:));
        meanS = mean(S(:)); stdS = std(S(:));
        meanV = mean(V(:)); stdV = std(V(:));

        features = [meanR, stdR, meanG, stdG, meanB, stdB, meanH, stdH, meanS, stdS, meanV, stdV];
    catch ME_hsv
        warning('Gagal konversi ke HSV atau ekstrak fitur HSV: %s. Menggunakan fitur RGB saja.', ME_hsv.message);
        features = [meanR, stdR, meanG, stdG, meanB, stdB, 0, 0, 0, 0, 0, 0]; % Beri nilai default jika HSV gagal
    end
end