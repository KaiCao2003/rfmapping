function RFmapping
    % RF files use indexed, compressed NPZ storage with a .rfmap extension.
    
    params.onlyReadGoodUnits = true;
    
    % only ONE true
    params.isBackgroundMoving = false;
    params.isAllocentricPixelBins = false;
    params.isRotation = false ;

    % Stimulus geometry. For vertical bars, set all three coordinate flags above false.
    params.isVerticalBar = true;
    params.barBinWidthDeg = 3;

    params.isFineResolution = false; % Applies only to egocentric, rotation, and allocentric pixel-bin maps.

    params.isUseRealCoordinate = true;
    
    % Run the python code to get on_list_time.npy and adc_spike_times.npy (not from pipeline)
    % CHANGE BELOW BEFORE EVERY RUN ahhhhh
    
    is_on = true;
    is_off = false;

    params.base_dir = '/mnt/senzailab/Kai/#Recording/m20/';
    params.date = '260918';
    params.probelist = 'A';

    params.sessionList = '3';

    % VS time window in seconds, relative to stimulus onset.
    % 
    % params.VSTimeWindow = [0 0.2];
    params.VSTimeWindow = [-0.1 0.4];
    timeBinWidthMs = 1;
   

    params.total_deg = 360;
    
    % input paras verification
    assert(sum([params.isBackgroundMoving, params.isAllocentricPixelBins, params.isRotation]) <= 1, ...
    'No more than one of the flags can be true');
    
    % UPDATES: always +1

    params.rotationOffsetSign = +1; 
    
    % % by default the both bg and animal rotation cw
    % expectedRotationOffsetSign = -1 * params.isBackgroundMoving + 1 * params.isRotation;
    % if expectedRotationOffsetSign ~= 0
    %     if expectedRotationOffsetSign ~= params.rotationOffsetSign
    %     fprintf('Rotation sign is not what expected. Please check..., the expected sign is %d\n', expectedRotationOffsetSign);
    %     else
    %         fprintf('Rotation sign is %d\n', params.rotationOffsetSign);
    %     end
    % end
    
    windowWidthMs = diff(params.VSTimeWindow) * 1000;
    nbinsExact = windowWidthMs / timeBinWidthMs;
    assert(abs(nbinsExact - round(nbinsExact)) < 1e-9, ...
        'VSTimeWindow width %.6g ms is not divisible by timeBinWidthMs %.6g ms.', ...
        windowWidthMs, timeBinWidthMs);

    params.nbins = round(nbinsExact);
    fprintf('time bin is %d ms/bin\n', timeBinWidthMs);
    fprintf('In total of %d bins\n', params.nbins);

    % 360 Screen info, do need to change. 
    params.screenWidthPix = 960;
    params.screenHeightPix = 240;
    params.screenDeg = 360;

    addpath(fullfile(fileparts(mfilename('fullpath')), 'Utils'));

    if is_on
        fprintf('----- Working on On RF ----- \n');
        params.lum = 1;
        RFmapping_core(params);
    end

    if is_off
        fprintf('----- Working on Off RF ----- \n');
        params.lum = 0;
        RFmapping_core(params);
    end


end
