function coarseMask = RFmapping_coarsen_trial_spatial_mask( ...
        nativeMask, yCount, nativeXCount, pixelsPerBin)

    trialCount = size(nativeMask, 1);
    coarseXCount = nativeXCount / pixelsPerBin;
    coarseBinCount = yCount * coarseXCount;
    trialIndexCell = cell(coarseBinCount, 1);
    coarseBinIndexCell = cell(coarseBinCount, 1);

    for coarseXIndex = 1:coarseXCount
        nativeXIndices = (coarseXIndex - 1) * pixelsPerBin + ...
            (1:pixelsPerBin);
        for yIndex = 1:yCount
            nativeBinIndices = yIndex + (nativeXIndices - 1) * yCount;
            qualifyingTrials = any(nativeMask(:, nativeBinIndices), 2);
            coarseBinIndex = yIndex + (coarseXIndex - 1) * yCount;
            trialIndexCell{coarseBinIndex} = find(qualifyingTrials);
            coarseBinIndexCell{coarseBinIndex} = coarseBinIndex * ...
                ones(nnz(qualifyingTrials), 1);
        end
    end

    trialIndex = vertcat(trialIndexCell{:});
    coarseBinIndex = vertcat(coarseBinIndexCell{:});
    coarseMask = sparse(trialIndex, coarseBinIndex, ...
        ones(numel(trialIndex), 1), ...
        trialCount, coarseBinCount);
end
