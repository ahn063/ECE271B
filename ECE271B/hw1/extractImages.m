function [folderImages, folderFilenames] = extractImages(mainDir)

    % Get a list of all files and folders in this folder
    files = dir(mainDir);

    % Initialize the outer cell arrays
    folderImages = {};
    folderFilenames = {};

    % Loop through them to find folders
    for k = 1:length(files)
        if files(k).isdir && ~strcmp(files(k).name, '.') && ~strcmp(files(k).name, '..')
            % It's a folder, construct the path to the folder
            folderPath = fullfile(mainDir, files(k).name);

            % Get a list of all images in this folder (assuming JPEG and PNG formats)
            imgFiles = dir(fullfile(folderPath, '*.jpg'));
            imgFiles = [imgFiles; dir(fullfile(folderPath, '*.png'))];

            % Initialize the inner cell arrays for this folder
            images = {};
            filenames = {};

            % Loop through each image
            for m = 1:length(imgFiles)
                % Construct the path to the image
                imgPath = fullfile(folderPath, imgFiles(m).name);

                % Read the image
                img = imread(imgPath);

                % Store the image and filename in the inner cell arrays
                images{end + 1} = img;
                filenames{end + 1} = imgFiles(m).name;
            end

            % Store the inner cell arrays in the outer cell arrays
            folderImages{end + 1} = images;
            folderFilenames{end + 1} = filenames;
        end
    end

    % folderImages now contains a cell array for each folder, each containing images
    % folderFilenames contains a corresponding cell array of filenames
end
