function groundTruth = loadYoloLabel(labelPath, imagePath)
% Function to obtain the ground truth of each image
   
    % Security check
    if ~isfile(labelPath)
        return;
    end

    % Read image
    img = imread(imagePath);
    if isempty(img)
        return;
    end

    % Get image real size
    [imgH, imgW, ~] = size(img);

    % Read YOLO label file (format: class_id xc yc w h)
    data = readmatrix(labelPath);

    if isempty(data)
        return;
    end

    % Initialize the output as empty struct array
    nObjects = size(data, 1);
    if nObjects == 0
        groundTruth = struct('box', {}, 'confidence', {}, 'className', {});
        return;
    end
    groundTruth(nObjects) = struct('box', [], 'confidence', [], 'className', []);
 
    % Iterate through all the objects in the file (there might be more than one)
    for i = 1:nObjects
        
        xc = data(i, 2);
        yc = data(i, 3);
        w = data(i, 4);
        h = data(i, 5);

        % Obtain the real dimensions of the bounding box (from normalized to pixel coordinates)
        boxW = int32(round(w * imgW));
        boxH = int32(round(h * imgH));

        % Obtain the coordenate of the upper left corner of the box (from normalized to pixel coordinates)
        x = int32(round((xc - w/2) * imgW));
        y = int32(round((yc - h/2) * imgH));

        % Create a ground truth structure 
        groundTruth(i).box = [x, y, boxW, boxH];
        groundTruth(i).confidence = 1.0;
        groundTruth(i).className = 'UAV';              
    end

    
end
