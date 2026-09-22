function RFmapping_core(params)
    onlyReadGoodUnits = params.onlyReadGoodUnits;
    isBackgroundMoving = params.isBackgroundMoving;
    isAllocentricPixelBins = params.isAllocentricPixelBins;
    isRotation = params.isRotation;
    rotationOffsetSign = params.rotationOffsetSign;
    isFineResolution = params.isFineResolution;
    isUseRealCoordinate = params.isUseRealCoordinate;
    isVerticalBar = params.isVerticalBar;
    barBinWidthDeg = params.barBinWidthDeg;
    date = params.date;
    probelist = params.probelist;
    sessionList = params.sessionList;
    lum = params.lum;

    total_deg = params.total_deg;
    screenWidthPix = params.screenWidthPix;
    screenHeightPix = params.screenHeightPix;
    screenDeg = params.screenDeg;
    
    VSTimeWindow = params.VSTimeWindow;
    nbins = params.nbins;

    assert(~isVerticalBar || ...
        ~(isBackgroundMoving || isAllocentricPixelBins || isRotation), ...
        'Vertical-bar mapping requires regular screen coordinates.');
    assert(~isVerticalBar || ...
        (screenWidthPix == 960 && screenHeightPix == 240 && ...
        screenDeg == 360 && total_deg == screenDeg), ...
        'Vertical-bar mapping requires a 960 x 240 pixel, 360 degree screen.');

    
   if lum == 1
        lumName = 'ON';
        lumSuffix = '';
        barColor = 'white';
    elseif lum == 0
        lumName = 'OFF';
        lumSuffix = '_off';
        barColor = 'black';
    elseif lum == 0.5
        lumName = 'GRAY';
        lumSuffix = '_gray';
        barColor = 'gray';
    end
    
    %%%%%%%%%%%%%%%%%%%%%
    % Process something before enter analysis
    %%%%%%%%%%%%%%%%%%%%%

    timeBinWidthMs = diff(VSTimeWindow) * 1000 / nbins;

    timeFolder = sprintf('%g_%g_%gms', ...
        VSTimeWindow(1) * 1000, VSTimeWindow(2) * 1000, timeBinWidthMs);
    if onlyReadGoodUnits
        unitSelectionFolder = 'good';
    else
        unitSelectionFolder = 'all';
    end

    pxPerDeg = screenWidthPix / screenDeg;
    screenCenterDeg = screenDeg / 2;
    
    fileExtension = '.rfmap';

    % Extract stim timing


    for session_raw = sessionList
        session = ['_', session_raw];

        sessionID = [date, session];
        
        fprintf('===========================\n');
        fprintf('Working on session: %s\n', sessionID);

        base_dir = [params.base_dir, date, '/', sessionID, '/'];
        
        
        trials_data_mat = [base_dir, date, '.mat'];
        trials_mat_adc = [base_dir, 'data/on_list_times.npy'];

        % load([fbasename '_Trials.mat']);
        trials = load(trials_data_mat).trials;
        trials_time = readNPY(trials_mat_adc);
        barCoverage = [];
        if isVerticalBar
            barCoverage = RFmapping_vertical_bar_coverage(trials, trials_time, ...
                screenWidthPix, screenDeg, barBinWidthDeg, lum);

            %%%%%% CHANGE %%%%%%%
            assert(all(barCoverage.nativePixelTrialCount > 0), ...
                ['Vertical-bar %s analysis requires %s-bar trials that ' ...
                'cover every horizontal screen pixel.'], lumName, barColor);
        end
        if isRotation
            hd_trials = [base_dir, 'data/hd_trials_times.npy'];
            hdOffsetPix = readNPY(hd_trials);
        end
        
        for probe = probelist
            fprintf('===========================\n')
            fprintf('Probe%s\n', probe);

            kilosort_folder = [base_dir, 'kilosort/Probe', probe, '/kilosort', session, '/'];
            % kilosort_folder = [base_dir, 'pipeline/260701/Probe', probe, '/kilosort'];
            clusterKSLabelFile = [kilosort_folder, 'cluster_KSLabel.tsv'];
            clusterGroupFile = [kilosort_folder, 'cluster_group.tsv'];
            spikeClustersFile = [kilosort_folder, 'spike_clusters.npy'];
            spikeTimesFile = [base_dir, 'data/probe', probe, '/adc_spike_time.npy'];

            %%%%%% CHANGE %%%%%%%
            save_dir = [base_dir, 'data/rfmapping', lumSuffix, '/', unitSelectionFolder, '/', timeFolder, '/Probe', probe, '/'];
        
            [spikeTimes, spikeClusters, clusterIds, clusterKSLabels] = ReadSpikeData( ...
                spikeTimesFile, spikeClustersFile, clusterKSLabelFile);
            
            goodUnits = clusterIds(strcmp(clusterKSLabels, 'good'));
        
            % DetectExtInputOnset(fbasename,'digital',1,'RFmapStim'); % includes both opto and visual
        
            % load([fbasename '_Pulses_RFmapStim.mat']);
        
            pulseNum = length(trials_time) - 1;
            
        
            VisStim = [];
            VisStim.periods = zeros(pulseNum, 2);
            VisStim.duration = zeros(pulseNum, 1);
            VisStim.PosX = zeros(pulseNum, 1);
            VisStim.PosY = zeros(pulseNum, 1);
            VisStim.SquareSize = zeros(pulseNum, 1);
            VisStim.Lum = zeros(pulseNum, 1);
            
            VisStim.periods = [trials_time(1:pulseNum), trials_time(2:pulseNum+1)];
            VisStim.duration = VisStim.periods(:,2) - VisStim.periods(:,1);
        
            for i = 1:pulseNum
                if isBackgroundMoving
        %%%%%egocentric%%%%%
                    posXPix = (trials(i).Square_PositionX + screenCenterDeg) * pxPerDeg;
                    offsetPix = trials(i).BackgroundRotation_XOffset_Pix;
                    
                    VisStim.PosX(i) = mod(posXPix + rotationOffsetSign * offsetPix, screenWidthPix);
        %%%%%egocentric%%%%%
                elseif isRotation
                    posXPix = (trials(i).Square_PositionX + screenCenterDeg) * pxPerDeg;
                    offsetPix = hdOffsetPix(i);
                    
                    VisStim.PosX(i) = mod(posXPix + rotationOffsetSign * offsetPix, screenWidthPix);
                else
                    VisStim.PosX(i) = trials(i).Square_PositionX;
                end
        
                VisStim.PosY(i) = trials(i).Square_PositionY;
                VisStim.SquareSize(i) = trials(i).Square_Size;
                VisStim.Lum(i) = trials(i).Square_Luminance;
        
            end
        
            % a = VisStim.duration;
            % 1/(mean(a(a>0.1)) - mean(a(a<0.1)))
        
            % SetCurrentSession('basename', fbasename);
            % unit_pool = GetUnits;
            
            unitPool = unique(spikeClusters);
            if onlyReadGoodUnits
                unitPool = intersect(unitPool, goodUnits);
            end
        
            unitNum = size(unitPool, 1);
            
        
            % elec = 1;
            % clu = load([fbasename '.clu.' num2str(elec)]);
            % clu = clu(2:end);
        
            xPositions = [];
            yPositions = [];
            if isVerticalBar
                SqDeg = barBinWidthDeg;
                xPositions = barCoverage.xPositionsDeg;
                yPositions = 0;
                squareWidthPix = [];
                squareWidthDeg = [];
                isPixelByPixelAnalysis = false;
            else
                SqDeg = VisStim.SquareSize(1);
                squareWidthPix = SqDeg * pxPerDeg;
                squareWidthDeg = SqDeg;
                isPixelByPixelAnalysis = isBackgroundMoving || isRotation || isAllocentricPixelBins;
                if isBackgroundMoving || isRotation
        %%%%%egocentric%%%%%
                    xPositions = 0:(screenWidthPix - 1);
                    yPositions = unique(VisStim.PosY);
                elseif isAllocentricPixelBins
                    allocentricBinNum = round(total_deg * pxPerDeg);
                    xPositions = -total_deg / 2 + (0:(allocentricBinNum - 1)) / pxPerDeg;
                    yPositions = unique(VisStim.PosY);
                else
                    actualXPositions = unique(VisStim.PosX).';
                    actualYPositions = unique(VisStim.PosY).';
                    if isUseRealCoordinate
                        xPositions = actualXPositions;
                        yPositions = actualYPositions;
                    else
                        x_num = length(actualXPositions);
                        y_num = length(actualYPositions);
                        xPositions = -(x_num - 1) / 2 * SqDeg + (0:x_num - 1) * SqDeg;
                        yPositions = -(y_num - 1) / 2 * SqDeg + (0:y_num - 1) * SqDeg;
                    end
                end
            end

            x_num = length(xPositions);
            y_num = length(yPositions);
            if isVerticalBar
                %%%%%% CHANGE %%%%%%%
                trialSpatialMask = RFmapping_coarsen_trial_spatial_mask( ...
                    barCoverage.trialPixelCoverage, 1, screenWidthPix, ...
                    barCoverage.pixelsPerBin);

                xPositionsForFile = xPositions;
            else
                %%%%%% CHANGE %%%%%%%
                trialSpatialMask = RFmapping_trial_spatial_mask(VisStim, ...
                    xPositions, yPositions, isBackgroundMoving || isRotation, ...
                    isAllocentricPixelBins, squareWidthPix, squareWidthDeg, ...
                    screenWidthPix, lum);

                if isPixelByPixelAnalysis && ~isFineResolution
                    regularXNum = total_deg / SqDeg;
                    pixelsPerRegularBin = screenWidthPix / regularXNum;
                    trialSpatialMask = RFmapping_coarsen_trial_spatial_mask( ...
                        trialSpatialMask, y_num, x_num, pixelsPerRegularBin);
                    x_num = regularXNum;
                    xPositionsForFile = -total_deg / 2 + SqDeg / 2 + ...
                        (0:(regularXNum - 1)) * SqDeg;
                else
                    xPositionsForFile = xPositions;
                end
            end
            occupancyTimeSec = reshape(VisStim.duration.' * trialSpatialMask, ...
                y_num, x_num);

            spikeTimesByUnit = RFmapping_group_spike_times(spikeTimes, spikeClusters, unitPool);
            RFmap = cell(unitNum, 1);
            parfor k = 1:unitNum
                s = spikeTimesByUnit{k};
                trialSpikeCounts = RFmapping_trial_spike_counts(s, ...
                    VisStim.periods(:, 1), VSTimeWindow, nbins);
                spatialSpikeCounts = full( ...
                    trialSpatialMask.' * trialSpikeCounts);
                unitRF = [];

                %%%%%% CHANGE %%%%%%%
                unitRF.(lumName).OnSet = reshape(spatialSpikeCounts, ...
                    y_num, x_num, nbins);
                RFmap{k} = unitRF;
                fprintf('done %d out of %d\n', k, unitNum);
            end

            rfmapMetadata = struct;
            rfmapMetadata.format = 'rfmap';
            rfmapMetadata.formatVersion = 2;
            rfmapMetadata.storage = 'indexed_npz';
            rfmapMetadata.unitArrayKeyPattern = 'unit_{unit_id}';
            rfmapMetadata.unitArrayAxes = {'y', 'x', 'time'};
            rfmapMetadata.unitsSpikeCountsSize = [unitNum, y_num, x_num, nbins];
            rfmapMetadata.VSTimeWindow = VSTimeWindow;
            rfmapMetadata.timeBinWidthMs = timeBinWidthMs;
            rfmapMetadata.isVerticalBar = isVerticalBar;
            rfmapMetadata.responseUnits = 'spike_count';
            rfmapMetadata.responseNormalization = 'none';
            if isVerticalBar
                rfmapMetadata.stimulusGeometry = 'vertical_bar_full_height';
                rfmapMetadata.eventDefinition = ...
                    ['trial_onset_for_each_covering_', barColor, '_bar'];
                rfmapMetadata.pixelTimingSource = ...
                    'on_list_times_trial_onsets_selected_by_renderer_footprint';
                rfmapMetadata.spatialAggregation = ...
                    'binary_final_bin_trial_coverage';
                rfmapMetadata.barWidthHandling = 'pooled';
                rfmapMetadata.mapInterpretation = ...
                    'coverage_conditioned_not_deconvolved';
                rfmapMetadata.barBinWidthDeg = barCoverage.binWidthDeg;
                rfmapMetadata.pixelsPerBarBin = barCoverage.pixelsPerBin;
                rfmapMetadata.nativePixelWidthDeg = screenDeg / screenWidthPix;
                rfmapMetadata.barWidthsDeg = barCoverage.barWidthsDeg;
                rfmapMetadata.barWidthsPix = barCoverage.barWidthsPix;
                rfmapMetadata.barWidthTrialCount = ...
                    barCoverage.barWidthTrialCount;
                rfmapMetadata.barHeightPix = screenHeightPix;
                rfmapMetadata.screenSizePix = [screenWidthPix, screenHeightPix];
                rfmapMetadata.screenWidthDeg = screenDeg;
                rfmapMetadata.nativePixelTrialCount = ...
                    barCoverage.nativePixelTrialCount;
                rfmapMetadata.nativePixelExposureSec = ...
                    barCoverage.nativePixelExposureSec;
                rfmapMetadata.nativePixelExposureDefinition = ...
                    'sum_of_qualifying_trial_durations';
                countByBin = permute(reshape( ...
                    barCoverage.nativePixelTrialCount, ...
                    barCoverage.pixelsPerBin, []), [2 1]);
                exposureByBin = permute(reshape( ...
                    barCoverage.nativePixelExposureSec, ...
                    barCoverage.pixelsPerBin, []), [2 1]);
                rfmapMetadata.meanNativePixelTrialCountByBin = ...
                    reshape(mean(countByBin, 2), 1, []);
                rfmapMetadata.nativePixelTrialCountRangeByBin = ...
                    [min(countByBin, [], 2), max(countByBin, [], 2)];
                rfmapMetadata.meanNativePixelExposureSecByBin = ...
                    reshape(mean(exposureByBin, 2), 1, []);
                rfmapMetadata.nativePixelExposureSecRangeByBin = ...
                    [min(exposureByBin, [], 2), max(exposureByBin, [], 2)];
            end
            
            % Validate the shared spatial axes before writing unit entries.
            assert(length(xPositionsForFile) == x_num);
            assert(isequal(size(occupancyTimeSec), [y_num, x_num]));

            if exist(save_dir, 'dir') ~= 7
                mkdir(save_dir);
            end
            if isVerticalBar
                rfmapFileName = sprintf( ...
                    'regular_unitsSpikeCounts_%s_vertical_bar_pooled_bin%gdeg%s%s', ...
                    sessionID, barBinWidthDeg, lumSuffix, fileExtension);
            elseif isBackgroundMoving
                if isFineResolution
                    rfmapFileName = ['egocentric_unitsSpikeCounts_', sessionID, lumSuffix, fileExtension];
                else
                    rfmapFileName = ['egocentric_30_unitsSpikeCounts_', sessionID, lumSuffix, fileExtension];
                end
            elseif isRotation
                if isFineResolution
                    rfmapFileName = ['rotation_unitsSpikeCounts_', sessionID, lumSuffix, fileExtension];
                else
                    rfmapFileName = ['rotation_30_unitsSpikeCounts_', sessionID, lumSuffix, fileExtension];
                end
            elseif isAllocentricPixelBins
                if isFineResolution
                    rfmapFileName = ['allocentric_pixelbins_unitsSpikeCounts_', sessionID, lumSuffix, fileExtension];
                else
                    rfmapFileName = ['allocentric_pixelbins_30_unitsSpikeCounts_', sessionID, lumSuffix, fileExtension];
                end
            else
                rfmapFileName = ['regular_unitsSpikeCounts_', sessionID, lumSuffix, fileExtension];
            end
            WriteRfmap(fullfile(save_dir, rfmapFileName), rfmapMetadata, ...
                unitPool, xPositionsForFile, yPositions, ...
                linspace(VSTimeWindow(1), VSTimeWindow(2), nbins + 1), ...
                occupancyTimeSec, RFmap, lumName);

        
            if ~isVerticalBar && ~isBackgroundMoving && ~isRotation && ...
                    ~isAllocentricPixelBins && ~isUseRealCoordinate
                totalResponse = 0;
                for k = 1:unitNum
                    totalResponse = totalResponse + sum(RFmap{k}.(lumName).OnSet(:));
                end
                if totalResponse == 0
                    fprintf('Suggest use the real y instead of precalculated due to the change of y axis by ptb\n');
                end
            end

        %%%%%egocentric%%%%%
            SaveRfPatternCsv(RFmap, unitPool, save_dir, total_deg, date, sessionID, ...
                isBackgroundMoving, isRotation, isAllocentricPixelBins && ~isBackgroundMoving, ...
                isFineResolution, isVerticalBar, barBinWidthDeg, barCoverage, ...
                screenWidthPix, screenHeightPix, occupancyTimeSec, ...
                lumName, lumSuffix, barColor);
        %%%%%egocentric%%%%%
        
        
            SaveRfPatternPdf(RFmap, unitPool, unitNum, save_dir, total_deg, ...
                isBackgroundMoving, isRotation, isAllocentricPixelBins, isFineResolution, ...
                isVerticalBar, barBinWidthDeg, isPixelByPixelAnalysis, lumName, lumSuffix);
        end
    end
end

function WriteRfmap(fileName, metadata, unitPool, xPositions, yPositions, ...
        timeBinEdges, occupancyTimeSec, RFmap, lumName)
    % NPZ schema: metadata is UTF-8 JSON in a uint8 NPY array; shared arrays
    % have their field names as keys. unit_<ID> stores float64 [y, x, time].
    % unitPool defines display order; ZIP indexes each unit independently.
    outputStream = java.io.BufferedOutputStream(java.io.FileOutputStream(fileName));
    archive = java.util.zip.ZipOutputStream(outputStream);
    archiveCleanup = onCleanup(@() archive.close());
    archive.setLevel(6);

    metadataBytes = unicode2native(jsonencode(metadata), 'UTF-8');
    WriteNpyEntry(archive, 'metadata', metadataBytes, '|u1', numel(metadataBytes));
    WriteNpyEntry(archive, 'unitPool', int64(unitPool), '<i8', numel(unitPool));
    WriteNpyEntry(archive, 'xPositions', xPositions, '<f8', numel(xPositions));
    WriteNpyEntry(archive, 'yPositions', yPositions, '<f8', numel(yPositions));
    WriteNpyEntry(archive, 'timeBinEdges', timeBinEdges, '<f8', numel(timeBinEdges));
    WriteNpyEntry(archive, 'occupancyTimeSec', occupancyTimeSec, '<f8', ...
        metadata.unitsSpikeCountsSize(2:3));

    for k = 1:numel(unitPool)
        WriteNpyEntry(archive, sprintf('unit_%d', unitPool(k)), ...
            RFmap{k}.(lumName).OnSet, '<f8', metadata.unitsSpikeCountsSize(2:4));
    end
    archive.close();
end

function WriteNpyEntry(archive, key, values, dtype, shape)
    % NPY 1.0 preserves MATLAB column order and explicit singleton dimensions.
    % Numeric payloads are little-endian on the supported MATLAB platforms.
    shapeText = sprintf('%d, ', shape);
    header = sprintf( ...
        '{''descr'': ''%s'', ''fortran_order'': True, ''shape'': (%s), }', ...
        dtype, shapeText);
    paddingLength = mod(-(10 + numel(header) + 1), 64);
    header = [header, repmat(' ', 1, paddingLength), newline];
    prefix = [uint8([147, double('NUMPY'), 1, 0]), ...
        typecast(uint16(numel(header)), 'uint8'), uint8(header)];

    archive.putNextEntry(java.util.zip.ZipEntry([key, '.npy']));
    archive.write(typecast(prefix, 'int8'), 0, numel(prefix));
    % Bound MATLAB-to-Java copies when a fine-resolution unit is large.
    elementsPerChunk = 131072;
    for first = 1:elementsPerChunk:numel(values)
        last = min(first + elementsPerChunk - 1, numel(values));
        bytes = typecast(values(first:last), 'int8');
        archive.write(bytes, 0, numel(bytes));
    end
    archive.closeEntry();
end
