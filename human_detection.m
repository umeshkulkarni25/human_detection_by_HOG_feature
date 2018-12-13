close all;
TRAIN_POSITIVE_FOLDER = 'Train_Positive';
TRAIN_NEGATIVE_FOLDER = 'Train_Negative';
TEST_POSITIVE_FOLDER = 'Test_Positive';
TEST_NEGATIVE_FOLDER = 'Test_Neg';
BASE_FOLDER = 'Human';

[positive_training_data] = collect_data([BASE_FOLDER filesep TRAIN_POSITIVE_FOLDER],1);
[negative_training_data] = collect_data([BASE_FOLDER filesep TRAIN_NEGATIVE_FOLDER],0);
training_data = [positive_training_data negative_training_data];
[training_data, feature_vector_length] = extract_features(training_data);
[network] = construct_mlp(feature_vector_length, 1000);
[network] = train_mlp(network, training_data);

[positive_testing_data] = collect_data([BASE_FOLDER filesep TEST_POSITIVE_FOLDER],1);
[negative_testing_data] = collect_data([BASE_FOLDER filesep TEST_NEGATIVE_FOLDER],0);
testing_data = [positive_testing_data negative_testing_data];
[testing_data, feature_vector_length] = extract_features(testing_data);
prdict_mlp(network, testing_data);

function prdict_mlp(network, testing_data)
    hidden_layer = network.hidden_layer;
    output_layer = network.output_layer;
    for iteration = 1: length(testing_data)
        testing_sample = testing_data(iteration);
        feature_vector = testing_sample.feature_vector;
        y1 = feature_vector * hidden_layer.w + hidden_layer.b;
        a1 = arrayfun(@(x) ReLU(x), y1);
        y2 = a1 * output_layer.w + output_layer.b;
        a2 = arrayfun(@(x) sigmoid(x), y2);
        disp(['should be ',num2str(testing_sample.label)]);
        disp(['prediction',num2str(a2)]);
    end
end

function [network] = construct_mlp(feature_vector_length, hidden_layer_size)
   network = struct;
   
   hidden_layer = struct;
   hidden_layer.w = rand(feature_vector_length,hidden_layer_size) .* 0.01;
   hidden_layer.b = rand(1, hidden_layer_size).* 0.01;
   network.hidden_layer = hidden_layer;
   
   output_layer = struct;
   output_layer.w = rand(hidden_layer_size,1).* 0.01;
   output_layer.b = rand(1, 1).* 0.01;
   network.output_layer = output_layer;
end

function [network] = train_mlp(network, training_data)
   hidden_layer = network.hidden_layer;
   output_layer = network.output_layer;
%    epoch_order = randperm(20);
   for epochs = 1:200
       errors = zeros(1,length(training_data));
       for iteration = 1: length(training_data)
        training_sample = training_data(iteration);
        feature_vector = training_sample.feature_vector;
        y1 = feature_vector * hidden_layer.w + hidden_layer.b;
        a1 = arrayfun(@(x) ReLU(x), y1);
        y2 = a1 * output_layer.w + output_layer.b;
        a2 = arrayfun(@(x) sigmoid(x), y2);

        errors(1, iteration) = (training_sample.label - a2)^2;
        gradient =  2 * (a2 - training_sample.label);
        gradient = arrayfun(@(x) delta_sigmoid(x), a2) .* gradient; 
        graidents_prev_layer = output_layer.w .* gradient;
        output_layer.w = output_layer.w - 0.01 .* a1' .*gradient;
        output_layer.b = output_layer.b - 0.01 .* gradient;
        graidents_prev_layer = arrayfun(@(x) delta_ReLU(x), y1').* graidents_prev_layer;
        hidden_layer.w = hidden_layer.w - 0.01 .* feature_vector' .* graidents_prev_layer';
        hidden_layer.b = hidden_layer.b - 0.01 .* graidents_prev_layer';  
       end
       disp(epochs)
       mean_error = sum(errors, 'all')/length(errors)
   end
end

function [a] = ReLU(x)
    if (x<=0)
        a = 0;
    else
        a = x;
    end
end

function [delta] = delta_ReLU(x)
    if (x<=0)
        delta = 0;
    else
        delta = 1;
    end
end

function [a] = sigmoid(x)
    a = 1/(1 + exp(-x));
end

function [delta] = delta_sigmoid(x)
    delta = x * (1-x);
end

function [data, feature_vector_length] = extract_features(data)
    for image_count = 1:length(data)
        data(image_count).feature_vector = extract_HOG_features(data(image_count).image);
    end
    feature_vector_length = length(data(1).feature_vector);
end

function [HOG_vector] = extract_HOG_features(image)
    [Gm, Ga] = apply_prewitt_operator(image);
    [cell_HOG] = get_cell_HOG(Gm, Ga);
    [HOG_vector] = concat_cell_HOGs(cell_HOG);
end
% function collects data from a folder named Human at the rot of the project %
function [data] = collect_data(folder, label)
    files = dir(fullfile(folder,'*.bmp'));
    data = struct([]);
    for file_count = 1:length(files)
        file = files(file_count,1);
        image = imread(fullfile(file.folder, file.name));
        data(file_count).image = get_grayscale_image(image);
        data(file_count).label = label;  
        data(file_count).fileName = file.name;
    end
end

% cuntion to concat cell-HOG into block HOG %
function [HOG_vector] = concat_cell_HOGs(cell_HOG)
    block_row_size = 2;
    block_column_size = 2;
    HOG_vector = zeros(1, (size(cell_HOG,1)-1) * (size(cell_HOG,2)-1) * (block_row_size*block_column_size*size(cell_HOG,3)));
    index = 1;
    for row = 1 :size(cell_HOG,1)-1
        for column = 1 :size(cell_HOG,2)-1
            block_HOG = cell_HOG(row:row+block_row_size-1, column:column+block_column_size-1, :); 
            normalized_block_HOG_vector = normalize_block_HOG(block_HOG);
            HOG_vector(index:index+size(normalized_block_HOG_vector,2)-1) = normalized_block_HOG_vector(:);
            index = index +  size(normalized_block_HOG_vector,2);
        end
    end
end

function [normalized_block_HOG_vector] = normalize_block_HOG(block_HOG)
  normalized_block_HOG_vector = zeros(1, numel(block_HOG));
  index = 1;
    for row = 1 :size(block_HOG,1)
        for column = 1:size(block_HOG,2)
            normalized_block_HOG_vector(index: index+size(block_HOG,3)-1) = block_HOG(row, column,:);
            index = index + size(block_HOG,3);
        end
    end
    L2_norm = sqrt(sum(normalized_block_HOG_vector.^2,'all'));
    if(L2_norm ~= 0)
        normalized_block_HOG_vector = normalized_block_HOG_vector ./ L2_norm;
    end 
end
% read the input image and convert it to grayscale %
function [gScale_image] = get_grayscale_image(color_image)
    % get R,G,B component of the image %
    R = double(color_image(:,:,1));
    G = double(color_image(:,:,2));
    B = double(color_image(:,:,3));
    % conversion and rounding off to produce grayscle image %
    gScale_image = double(0.299 .* R + 0.587 .* G + 0.114 .* B); 
end

function [HOG] = get_cell_HOG(Gm, Ga)
    cellRowSize = 8; cellColumnSize = 8;
    bin_centers = [0, 20, 40, 60, 80, 100, 120, 140, 160];
    HOG = zeros(size(Ga,1)/cellRowSize, size(Ga,2)/cellRowSize,size(bin_centers,2));
    for row = 1: cellRowSize :size(Ga,1) 
        for column = 1: cellColumnSize :size(Ga,2)
            gradient_cell = Ga(row:row+cellRowSize-1, column:column+cellColumnSize-1); 
            mangitude_cell = Gm(row:row+cellRowSize-1, column:column+cellColumnSize-1); 
            cell_HOG = calc_local_HOG(mangitude_cell,gradient_cell,bin_centers);
            HOG((row+cellRowSize-1)/cellRowSize,(column+cellColumnSize-1)/cellColumnSize,:) = cell_HOG(:);
        end
    end
end

function [localHistogram] = calc_local_HOG(mangitude_cell, gradient_cell, bin_centers)
    dist_bet_centers = 20;
    localHistogram = zeros(size(bin_centers));
    for row = 1:size(gradient_cell,1)
        for column = 1:size(gradient_cell,2)
            gradient = gradient_cell(row, column);
            magnitude = mangitude_cell(row, column);
            bin_1 = mod(floor(gradient / dist_bet_centers),size(bin_centers,2))+1;
            bin_2 = mod(ceil(gradient / dist_bet_centers),size(bin_centers,2))+1;
            bin2_dist = abs(bin_centers(bin_2) - gradient);
            bin1_dist = abs(bin_centers(bin_1) - gradient);
            if(bin2_dist > dist_bet_centers) 
                bin2_dist = 180 - bin2_dist;
            end
            if(bin1_dist > dist_bet_centers) 
                bin1_dist = 180 - bin1_dist;
            end
            localHistogram(bin_1) = localHistogram(bin_1) + bin2_dist/dist_bet_centers * magnitude;
            localHistogram(bin_2) =  localHistogram(bin_2) + bin1_dist/dist_bet_centers * magnitude;
        end
    end
end

% function to apply prewitt_operator to provided image and 
% return gradient magnitude and gradient angle matrices %
function [Gm, Ga] = apply_prewitt_operator(image)
    prewitt_operator_Gx = ([
        -1,  0,  1;
        -1,  0,  1;
        -1,  0,  1
        ]);
    prewitt_operator_Gy = ([
        1,   1,   1;
        0,   0,   0;
       -1,  -1,  -1;
        ]);

    % calculate horizontal gradient %
    Gx = convolution(image, prewitt_operator_Gx);
    
    % normalize and display the horizontal gradient image %
    normalized_Gx = Gx(:,:);
    for row = 1:size(normalized_Gx,1)
        for column = 1:size(normalized_Gx,2)
            pixel_value = normalized_Gx(row, column);
            if pixel_value < 0 
                pixel_value = -1 * pixel_value;
            end
            normalized_Gx(row, column) = pixel_value / 3;
        end
    end
    fig6 = figure('Name', 'horizontal-gradient image','NumberTitle', 'off');
   % imshow(uint8(normalized_Gx)); 
%     saveas(fig6, 'horizontal_gradient.bmp');
    
    % calculate vertical gradient %
    Gy = convolution(image, prewitt_operator_Gy);
    
    % normalize and display the horizontal gradient image %
    normalized_Gy = Gy(:,:);
    for row = 1:size(normalized_Gy,1)
        for column = 1:size(normalized_Gy,2)
            pixel_value = normalized_Gy(row, column);
            if pixel_value < 0 
                pixel_value = -1 * pixel_value;
            end
            normalized_Gy(row, column) = pixel_value / 3;
        end
    end
    fig7 = figure('Name','vertical-gradient image', 'NumberTitle', 'off');
   % imshow(uint8(normalized_Gy)); 
%     saveas(fig7, 'vertical_gradient.bmp');

    % calculate gradient maginutde image %
    Gm = zeros(size(Gx));
    for row = 1:size(Gm ,1)
        for column = 1:size(Gm,2)
           Gm(row, column) = sqrt(Gx(row, column)^2+Gy(row, column)^2);
        end
    end
    
    % normalize and show gradient magnitude image %
    normalized_Gm = Gm./(sqrt(2)*3);
    fig8 = figure('Name','normalized gradient magnitude image', 'NumberTitle', 'off'); imshow(uint8(normalized_Gm)); 
    saveas(fig8,'gradient_magnitude.bmp');
    % calculate gradient angles %
    Ga = atan2d(Gy,Gx);
    Ga(Ga >= 170) = Ga(Ga >= 170) - 180;
    Ga(Ga < -10) = Ga(Ga < -10) + 180;
end

% function to convolve any provided image with any provided filter %
function [filtered_image] = convolution (image, filter)
    filtered_image = image(:,:);
    % calulate from and to indices for the convolution operation based on 
    % provided filter size and image size %
    [filterRow, filterCol] = size(filter);
    [imageRow, imageCol] = size(filtered_image);
    convolutionRowStart = (filterRow + 1)/2;
    convolutionRowEnd = (imageRow - (filterRow - 1)/2);
    convolutionColStart = (filterCol + 1)/2;
    convolutionColEnd = (imageCol - (filterCol - 1)/2);
    for row = convolutionRowStart:convolutionRowEnd
        for col = convolutionColStart: convolutionColEnd
            filtered_image(row, col) = getWeightedSum(image, row, col, filter); 
        end
    end
    filtered_image = filtered_image(convolutionRowStart:convolutionRowEnd,convolutionColStart: convolutionColEnd);
    filtered_image = padarray(filtered_image,[1 1],'both');
end

% function to multiply the filter window with image at 
% a provided point of reference %
function [weightedSum] = getWeightedSum(image, refRow, refCol, filter)
    weightedSum = 0;
    [filterRow, filterCol] = size(filter);
    refRow = refRow - (filterRow-1)/2;
    refCol = refCol - (filterCol-1)/2;
    for row = 1:filterRow
        for col = 1: filterCol
            imRow = refRow+row-1;
            imCol = refCol+col-1;
            weightedSum = weightedSum + filter(row, col)*image(imRow,imCol);
        end
    end
end