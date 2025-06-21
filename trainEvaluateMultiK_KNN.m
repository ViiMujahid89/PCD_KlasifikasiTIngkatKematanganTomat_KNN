    clc;
    clear;
    close all;
        
    datasetPath = 'datasettomat/'; 
    imageSize = [100 100];         
    useNormalization = true;       
    k_values_to_test = [1, 3, 5, 7, 9, 11]; 
    k_value_final_for_saving = 5; 
    
    if ~isfolder(datasetPath)
        errorMessage = sprintf('Error: Folder dataset "%s" tidak ditemukan di dalam folder proyek Anda.\nPastikan path sudah benar.', datasetPath);
        uiwait(errordlg(errorMessage, 'Kesalahan Path Dataset', 'modal'));
        return;
    end
    
    imds = imageDatastore(datasetPath, ...
        'IncludeSubfolders',true, ...
        'LabelSource','foldernames');
    
    numClasses = numel(categories(imds.Labels));
    labelNames = categories(imds.Labels); 
    
    allFeatures = [];
    allLabelsNumeric = zeros(numel(imds.Files), 1); 
    paramsForNormalization = struct(); 
    
    disp('Memulai preprocessing dan ekstraksi fitur...');
    for i = 1:numel(imds.Files)
        imgPath = imds.Files{i};
        imgOriginal = imread(imgPath);
    
        if size(imgOriginal,3) == 1
            imgOriginal = cat(3, imgOriginal, imgOriginal, imgOriginal); 
        elseif size(imgOriginal,3) == 4 
            imgOriginal = imgOriginal(:,:,1:3); 
        end
        imgResized = imresize(imgOriginal, imageSize); 
    
        currentFeatures = extractColorFeaturesTomato(imgResized);
        allFeatures = [allFeatures; currentFeatures];
    
        currentLabelName = imds.Labels(i);
        allLabelsNumeric(i) = find(strcmp(labelNames, char(currentLabelName)));
    
        if mod(i, 50) == 0 
            fprintf('Memproses gambar %d/%d\n', i, numel(imds.Files));
        end
    end
    disp('Preprocessing dan ekstraksi fitur selesai.');
    
    X_data_to_use = allFeatures; 
    if useNormalization
        disp('Melakukan normalisasi fitur (min-max ke rentang [0, 1])...');
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
        disp('Menggunakan fitur asli (tanpa normalisasi).');
    end
    
    disp('Membagi data latih dan data uji (70% latih, 30% uji)...');
    rng('default'); 
    cv = cvpartition(allLabelsNumeric, 'HoldOut', 0.3); 
    idxTrain = training(cv);
    idxTest = test(cv);
    % Buat set data latih dan uji
    XTrain = X_data_to_use(idxTrain,:);
    YTrain = allLabelsNumeric(idxTrain,:);
    XTest = X_data_to_use(idxTest,:);
    YTest = allLabelsNumeric(idxTest,:);
    fprintf('Jumlah data latih: %d, Jumlah data uji: %d\n', numel(YTrain), numel(YTest));
    
    disp('--- Menguji K-NN dengan Berbagai Nilai K ---');
    num_k_tests = numel(k_values_to_test);
    accuracies_train = zeros(1, num_k_tests);
    accuracies_test = zeros(1, num_k_tests); 
    k_labels_for_plot = cell(1, num_k_tests);
    knnModel_final_to_save = [];
    
    if license('test', 'Statistics_Toolbox') 
        for k_idx = 1:num_k_tests
            current_k = k_values_to_test(k_idx);
            k_labels_for_plot{k_idx} = ['K=', num2str(current_k)];
            fprintf('\nMelatih & menguji K-NN dengan K = %d...\n', current_k);
            try
                knnModel_current = fitcknn(XTrain, YTrain, 'NumNeighbors', current_k, 'Distance', 'euclidean');
    
                YPred_Train = predict(knnModel_current, XTrain);
                accuracies_train(k_idx) = sum(YPred_Train == YTrain) / numel(YTrain) * 100;
                fprintf('Akurasi Data Latih (K=%d): %.2f%%\n', current_k, accuracies_train(k_idx));
    
                YPred_Test = predict(knnModel_current, XTest);
                accuracies_test(k_idx) = sum(YPred_Test == YTest) / numel(YTest) * 100;
                fprintf('Akurasi Data Uji (K=%d): %.2f%%\n', current_k, accuracies_test(k_idx));
    
                if current_k == k_value_final_for_saving
                    knnModel_final_to_save = knnModel_current;
                end
            catch ME_knn_loop
                warning('Error saat melatih atau menguji K-NN dengan K=%d: %s', current_k, ME_knn_loop.message);
                accuracies_train(k_idx) = NaN; % Tandai sebagai tidak valid jika error
                accuracies_test(k_idx) = NaN;
            end
        end
    
        figure('Name', 'Perbandingan Akurasi K-NN vs Nilai K', 'NumberTitle', 'off');
        subplot(2,1,1);
        bar(categorical(k_labels_for_plot), accuracies_train, 'FaceColor', [0.2 0.6 0.8]);
        title('Akurasi K-NN pada Data Latih vs Nilai K');
        ylabel('Akurasi (%)');
        ylim([max(0, min(accuracies_train(~isnan(accuracies_train)))-10) 100.5]);
        grid on;
        text(1:num_k_tests, accuracies_train, sprintfc('%.1f%%', accuracies_train), 'Horiz','center', 'Vert','bottom', 'FontSize', 8, 'Color','black');
    
        subplot(2,1,2);
        bar(categorical(k_labels_for_plot), accuracies_test, 'FaceColor', [0.8 0.6 0.2]);
        title('Akurasi K-NN pada Data Uji vs Nilai K');
        ylabel('Akurasi (%)');
        xlabel('Jumlah Tetangga Terdekat (K)');
        ylim([max(0, min(accuracies_test(~isnan(accuracies_test)))-10) max(100.5, max(accuracies_test(~isnan(accuracies_test)))+5) ]); % Atur limit sumbu y
        grid on;
        text(1:num_k_tests, accuracies_test, sprintfc('%.1f%%', accuracies_test), 'Horiz','center', 'Vert','bottom', 'FontSize', 8, 'Color','black');
    
        if ~isempty(knnModel_final_to_save)
            YPred_Final_Test = predict(knnModel_final_to_save, XTest);
            accuracy_final_test = sum(YPred_Final_Test == YTest) / numel(YTest) * 100;
    
            figure('Name', ['Evaluasi Model K-NN Final (K=', num2str(k_value_final_for_saving), ')'], 'NumberTitle', 'off');
            YTestCategorical = categorical(YTest, 1:numClasses, labelNames, 'Ordinal',true);
            YPredFinalCategorical = categorical(YPred_Final_Test, 1:numClasses, labelNames, 'Ordinal',true);
    
            confusionchart(YTestCategorical, YPredFinalCategorical, ...
                           'RowSummary','row-normalized', ...
                           'ColumnSummary','column-normalized', ...
                           'Title', sprintf('Matriks Konfusi K-NN (K=%d, Akurasi Uji=%.2f%%)', k_value_final_for_saving, accuracy_final_test));
    
            save('tomato_maturity_knn_model.mat', ...
                 'knnModel_final_to_save', 'labelNames', 'imageSize', ...
                 'k_value_final_for_saving', 'numClasses', ...
                 'useNormalization', 'paramsForNormalization');
            disp(['Model K-NN final dengan K=', num2str(k_value_final_for_saving), ' telah disimpan ke file tomato_maturity_knn_model.mat']);
        else
            disp(['Model K-NN final dengan K=', num2str(k_value_final_for_saving), ' tidak berhasil dilatih atau disimpan.']);
        end
    else
        uiwait(errordlg('Statistics and Machine Learning Toolbox tidak ditemukan. Proses K-NN tidak dapat dilanjutkan.', 'Toolbox Error', 'modal'));
    end
    disp('Proses pelatihan dan evaluasi selesai.');