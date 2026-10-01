function [spikeTimes, spikeClusters, clusterIds, clusterKSLabels] = ReadSpikeData( ...
        spikeTimesFile, spikeClustersFile, clusterKSLabelFile)
    labels = readtable(clusterKSLabelFile, 'FileType', 'text', 'Delimiter', '\t');
    spikeClusters = readNPY(spikeClustersFile);
    spikeTimes = readNPY(spikeTimesFile);
    clusterIds = labels.cluster_id;
    clusterKSLabels = labels.KSLabel;
end
