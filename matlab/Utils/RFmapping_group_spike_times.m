function spikeTimesByUnit = RFmapping_group_spike_times(spikeTimes, spikeClusters, unitPool)
    % Keep acquisition order within each unit while scanning cluster IDs once.
    [selected, unitIndices] = ismember(spikeClusters(:), unitPool(:));
    selectedTimes = spikeTimes(selected);
    unitIndices = unitIndices(selected);
    [~, order] = sort(unitIndices);
    unitCounts = accumarray(unitIndices, 1, [numel(unitPool), 1]);
    spikeTimesByUnit = mat2cell(reshape(selectedTimes(order), [], 1), unitCounts);
end
