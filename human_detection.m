close all;
gScale_image = get_grayscale_image('./Human/Train_Positive/crop001030c.bmp');
gScale_image = double(gScale_image);
[Gm, Ga] = apply_prewitt_operator(gScale_image);
[cell_HOG] = get_cell_HOG(Gm, Ga);
[HOG_vector] = concat_cell_HOGs(cell_HOG);

% cuntion to concat cell-HOG into block HOG %
function [HOG_vector] = concat_cell_HOGs(cell_HOG)
    block_row_size = 2;
    block_column_size = 2;
    HOG_vector = zeros(1, (size(cell_HOG,1)-1) * (size(cell_HOG,2)-1) * (block_row_size*block_column_size*size(cell_HOG,3)));
    index = 1;
    for row = 1: block_row_size :size(cell_HOG,1)
        for column = 1: block_column_size :size(cell_HOG,2)
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
    normalized_block_HOG_vector = normalized_block_HOG_vector ./L2_norm;
end
% read the input image and convert it to grayscale %
function [gScale_image] = get_grayscale_image(image_path)
    original_image = imread(image_path);
    % get R,G,B component of the image %
    R = original_image(:,:,1);
    G = original_image(:,:,2);
    B = original_image(:,:,3);
    % conversion and rounding off to produce grayscle image %
    gScale_image = round((0.299 .* R + 0.587 .* G + 0.114 .* B)); 
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
            localHistogram(bin_1) = localHistogram(bin_1) + abs(bin_centers(bin_2) - gradient)/dist_bet_centers * magnitude;
            localHistogram(bin_2) =  localHistogram(bin_2) + abs(bin_centers(bin_1) - gradient)/dist_bet_centers * magnitude;
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
    imshow(uint8(normalized_Gx)); 
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
    imshow(uint8(normalized_Gy)); 
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
    Ga(Ga > 170) = Ga(Ga >170) - 180;
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