function trialSpatialMask = RFmapping_trial_spatial_mask(VisStim, ...
        xPositions, yPositions, isMovingCoordinates, ...
        isAllocentricPixelBins, squareWidthPix, squareWidthDeg, ...
        screenWidthPix, luminance)

    trialCount = numel(VisStim.duration);
    xCount = numel(xPositions);
    yCount = numel(yPositions);
    spatialBinCount = yCount * xCount;
    trialIndexCell = cell(spatialBinCount, 1);
    spatialBinIndexCell = cell(spatialBinCount, 1);

    for xIndex = 1:xCount
        for yIndex = 1:yCount
            currentX = xPositions(xIndex);
            currentY = yPositions(yIndex);
            if isMovingCoordinates
                dx = mod(currentX - VisStim.PosX + screenWidthPix / 2, ...
                    screenWidthPix) - screenWidthPix / 2;
                coversBin = dx >= -squareWidthPix / 2 & ...
                    dx < squareWidthPix / 2;
            elseif isAllocentricPixelBins
                dx = currentX - VisStim.PosX;
                coversBin = dx >= -squareWidthDeg / 2 & ...
                    dx < squareWidthDeg / 2;
            else
                coversBin = VisStim.PosX == currentX;
            end
            qualifyingTrials = coversBin & VisStim.PosY == currentY & ...
                VisStim.Lum == luminance;
            spatialBinIndex = yIndex + (xIndex - 1) * yCount;
            trialIndexCell{spatialBinIndex} = find(qualifyingTrials);
            spatialBinIndexCell{spatialBinIndex} = spatialBinIndex * ...
                ones(nnz(qualifyingTrials), 1);
        end
    end

    trialIndex = vertcat(trialIndexCell{:});
    spatialBinIndex = vertcat(spatialBinIndexCell{:});
    trialSpatialMask = sparse(trialIndex, spatialBinIndex, ...
        ones(numel(trialIndex), 1), ...
        trialCount, spatialBinCount);
end
