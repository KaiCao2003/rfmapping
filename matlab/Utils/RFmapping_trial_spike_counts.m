function spikeCount = RFmapping_trial_spike_counts(spikeTimes, eventTimes, ...
        VSTimeWindow, nbins)

    eventTimes = double(eventTimes(:));
    spikeCount = zeros(numel(eventTimes), nbins);
    [synchronized, eventIndex] = Sync(spikeTimes, eventTimes, ...
        'durations', VSTimeWindow);
    if isempty(synchronized)
        return;
    end

    timeEdges = linspace(VSTimeWindow(1), VSTimeWindow(2), nbins + 1);
    timeBin = discretize(synchronized(:, 1), timeEdges);
    isInside = ~isnan(timeBin) & synchronized(:, 1) < VSTimeWindow(2);
    spikeCount = accumarray([eventIndex(isInside), timeBin(isInside)], 1, ...
        [numel(eventTimes), nbins]);
end
