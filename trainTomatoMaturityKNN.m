% trainTomatoMaturityKNN.m

clc;
clear;
close all;

datasetPath = 'dataset/';
imageSize = [100 100];

if ~isfolder(datasetPath)
    errorMessage = sprintf('Error: Folder dataset tidak ditemukan di path:\n%s\n\nPastikan path sudah benar dan folder dataset ada.', datasetPath);
    uiwait(errordlg(errorMessage, 'Kesalahan Path Dataset', 'modal'));
    return;
end

maturityFolders = dir(datasetPath);
maturityFolders = maturityFolders([maturityFolders.isdir] & ~ismember({maturityFolders.name},{'.','..'}));

if isempty(maturityFolders)
    errorMessage = sprintf('Error: Tidak ada folder kelas kematangan yang ditemukan di dalam:\n%s', datasetPath);
    uiwait(errordlg(errorMessage, 'Kesalahan Struktur Dataset', 'modal'));
    return;
end

numClasses = numel(maturityFolders);
allFeatures = [];
allLabels = [];
labelNames = cell(1, numClasses);
paramsForNormalization = struct();
useNormalization = true;

disp('Memulai proses ekstraksi fitur...');
for i = 1:numClasses
    className = maturityFolders(i).name;
    labelNames{i} = strrep(className, '_', ' ');
    currentFolder = fullfile(datasetPath, className);
    
    imageList = [dir(fullfile(currentFolder, '*.jpg')); ...
                 dir(fullfile(currentFolder, '*.png')); ...
                 dir(fullfile(currentFolder, '*.jpeg')); ...
                 dir(fullfile(currentFolder, '*.JPG')); ...
                 dir(fullfile(currentFolder, '*.PNG')); ...
                 dir(fullfile(currentFolder, '*.JPEG'))];
    
    if ~isempty(imageList)
        [~, uniqueIdx] = unique({imageList.name});
        imageList = imageList(uniqueIdx);
    end

    if isempty(imageList)
        warning('Tidak ada file gambar yang ditemukan di folder: %s', className);
        continue;
    end

    fprintf('Memproses kelas: %s (%d gambar)\n', labelNames{i}, numel(imageList));

    for j = 1:numel(imageList)
        imgPath = fullfile(currentFolder, imageList(j).name);
        try
            img = imread(imgPath);

            if size(img,3) == 1
                img = cat(3, img, img, img);
            elseif size(img,3) == 4
                img = img(:,:,1:3);
            end
            imgResized = imresize(img, imageSize);

            currentFeatures = extractColorFeaturesTomato(imgResized); 
            
            allFeatures = [allFeatures; currentFeatures];
            allLabels = [allLabels; i];
        catch ME
            warning('Gagal memproses gambar %s: %s. Dilewati.', imgPath, ME.message);
        end
    end
end

if isempty(allFeatures) || isempty(allLabels)
    uiwait(errordlg('Ekstraksi fitur gagal atau tidak ada data yang berhasil diekstrak. Periksa dataset.', 'Kegagalan Ekstraksi Fitur', 'modal'));
    return;
end
disp('Ekstraksi fitur selesai.');
fprintf('Total fitur diekstrak: %d dari %d gambar.\n', size(allFeatures, 1), numel(allLabels));

X_data_to_use = allFeatures;

if useNormalization
    disp('Melakukan normalisasi fitur (min-max ke [0, 1])...');
    minVals = min(allFeatures, [], 1);
    maxVals = max(allFeatures, [], 1);
    rangeVals = maxVals - minVals;
    rangeVals(rangeVals == 0) = 1;
    
    normalizedFeatures = (allFeatures - minVals) ./ rangeVals;
    X_data_to_use = normalizedFeatures;
    
    paramsForNormalization.minVals = minVals;
    paramsForNormalization.rangeVals = rangeVals;
    disp('Normalisasi fitur selesai.');
else
    paramsForNormalization = [];
    disp('Menggunakan fitur asli (tanpa normalisasi).');
end


disp('Membagi data latih dan data uji...');
rng('default');
cv = cvpartition(allLabels, 'HoldOut', 0.3);
idxTrain = training(cv);
idxTest = test(cv);

XTrain = X_data_to_use(idxTrain,:);
YTrain = allLabels(idxTrain,:);
XTest = X_data_to_use(idxTest,:);
YTest = allLabels(idxTest,:);

if isempty(XTrain) || isempty(XTest)
    uiwait(errordlg('Gagal membagi dataset. Jumlah data mungkin terlalu sedikit.', 'Kegagalan Pembagian Data', 'modal'));
    return;
end
fprintf('Jumlah data latih: %d\n', size(XTrain, 1));
fprintf('Jumlah data uji: %d\n', size(XTest, 1));


disp('Melatih model KNN...');
k_value = 5;

if license('test', 'Statistics_Toolbox')
    try
        knnModel = fitcknn(XTrain, YTrain, 'NumNeighbors', k_value, 'Distance', 'euclidean');
        disp('Model KNN berhasil dilatih.');

        YPred_KNN = predict(knnModel, XTest);
        accuracy_KNN = sum(YPred_KNN == YTest) / numel(YTest) * 100;
        fprintf('Akurasi Model KNN pada data uji: %.2f%%\n', accuracy_KNN);

        allPossibleNumericClasses = 1:numClasses;
        YTestCategorical = categorical(YTest, allPossibleNumericClasses, labelNames, 'Ordinal', true);
        YPredCategorical_KNN = categorical(YPred_KNN, allPossibleNumericClasses, labelNames, 'Ordinal', true);

        figure('Name', 'Evaluasi Model KNN');
        confusionchart(YTestCategorical, YPredCategorical_KNN, ...
                       'RowSummary','row-normalized', ...
                       'ColumnSummary','column-normalized', ...
                       'Title', sprintf('Matriks Konfusi KNN (K=%d, Akurasi=%.2f%%)', k_value, accuracy_KNN));
        
        save('tomato_knn_model.mat', 'knnModel', 'labelNames', 'imageSize', 'k_value', 'numClasses', 'paramsForNormalization', 'useNormalization');
        disp('Model KNN dan informasi pendukung telah disimpan ke tomato_knn_model.mat');

    catch ME_knn
        uiwait(errordlg(sprintf('Error saat melatih atau mengevaluasi KNN: %s', ME_knn.message), 'Error Pelatihan KNN', 'modal'));
    end
else
    uiwait(errordlg('Statistics and Machine Learning Toolbox tidak ditemukan. Tidak dapat melatih KNN.', 'Toolbox Error', 'modal'));
end

disp('Proses pelatihan dan evaluasi KNN selesai.');


