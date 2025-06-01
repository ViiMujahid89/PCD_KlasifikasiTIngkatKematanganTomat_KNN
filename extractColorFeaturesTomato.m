function features = extractColorFeaturesTomato(img)

    R = double(img(:,:,1));
    G = double(img(:,:,2));
    B = double(img(:,:,3));

    meanR = mean(R(:)); stdR = std(R(:));
    meanG = mean(G(:)); stdG = std(G(:));
    meanB = mean(B(:)); stdB = std(B(:));

    try
        imgHSV = rgb2hsv(img);
        H = double(imgHSV(:,:,1));
        S = double(imgHSV(:,:,2));
        V = double(imgHSV(:,:,3));

        meanH = mean(H(:)); stdH = std(H(:));
        meanS = mean(S(:)); stdS = std(S(:));
        meanV = mean(V(:)); stdV = std(V(:));
        
        features = [meanR, stdR, meanG, stdG, meanB, stdB, meanH, stdH, meanS, stdS, meanV, stdV];
    catch ME_hsv
        warning('Gagal konversi ke HSV atau ekstrak fitur HSV: %s. Menggunakan fitur RGB saja.', ME_hsv.message);
        features = [meanR, stdR, meanG, stdG, meanB, stdB, 0, 0, 0, 0, 0, 0];
    end
end