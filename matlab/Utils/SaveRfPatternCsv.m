% Build the csv file for python

function SaveRfPatternCsv(RFmap, unitPool, save_dir, total_deg, date, sessionID, ...
        isBackgroundMoving, isRotation, isAllocentricPixelBins, isFineResolution, ...
        isVerticalBar, barBinWidthDeg, barCoverage, screenWidthPix, screenHeightPix, ...
        occupancyTimeSec, lumName, lumSuffix, barColor)
    
    if isVerticalBar
        csvFile = fullfile(save_dir, sprintf( ...
            'regular_%s_vertical_bar_pooled_bin%gdeg%s.csv', ...
            sessionID, barBinWidthDeg, lumSuffix));
    elseif isBackgroundMoving
        if isFineResolution
            csvFile = fullfile(save_dir, ['egocentric_', sessionID, lumSuffix, '.csv']);
        else
            csvFile = fullfile(save_dir, ['egocentric_30_', sessionID, lumSuffix, '.csv']);
        end
    elseif isRotation
        if isFineResolution
            csvFile = fullfile(save_dir, ['rotation_', sessionID, lumSuffix, '.csv']);
        else
            csvFile = fullfile(save_dir, ['rotation_30_', sessionID, lumSuffix, '.csv']);
        end
    elseif isAllocentricPixelBins
        if isFineResolution
            csvFile = fullfile(save_dir, ['allocentric_pixelbins_', sessionID, lumSuffix, '.csv']);
        else
            csvFile = fullfile(save_dir, ['allocentric_pixelbins_30_', sessionID, lumSuffix, '.csv']);
        end
    else
        csvFile = fullfile(save_dir, ['regular_', sessionID, lumSuffix, '.csv']);
    end
    csvDir = fileparts(csvFile);
    if exist(csvDir, 'dir') ~= 7
        mkdir(csvDir);
    end

    innerBlankRows = 4;
    polarPadRows = 1;
    unitNum = length(unitPool);
    sampleRf = flipud(sum(RFmap{1}.(lumName).OnSet, 3));
    occupancyMap = flipud(occupancyTimeSec);
    [rfRows, rfCols] = size(sampleRf);
    rowCount = unitNum * rfRows * rfCols;

    unit_index = zeros(rowCount, 1);
    unit_id = zeros(rowCount, 1);
    row = zeros(rowCount, 1);
    col = zeros(rowCount, 1);
    rf_value = zeros(rowCount, 1);
    occupancy_time_sec = zeros(rowCount, 1);
    theta_start_deg = zeros(rowCount, 1);
    theta_end_deg = zeros(rowCount, 1);
    r_inner = zeros(rowCount, 1);
    r_outer = zeros(rowCount, 1);
    rf_rows = rfRows * ones(rowCount, 1);
    rf_cols = rfCols * ones(rowCount, 1);
    total_deg_col = total_deg * ones(rowCount, 1);
    inner_blank_rows = innerBlankRows * ones(rowCount, 1);
    polar_plot_radius = (innerBlankRows + rfRows + polarPadRows) * ones(rowCount, 1);

    thetaEdges = linspace(90 + total_deg / 2, 90 - total_deg / 2, rfCols + 1);
    rEdges = innerBlankRows:(innerBlankRows + rfRows);

    idx = 1;
    for k = 1:unitNum
        rf = flipud(sum(RFmap{k}.(lumName).OnSet, 3));
        for rr = 1:rfRows
            for cc = 1:rfCols
                unit_index(idx) = k;
                unit_id(idx) = unitPool(k);
                row(idx) = rr;
                col(idx) = cc;
                rf_value(idx) = rf(rr, cc);
                occupancy_time_sec(idx) = occupancyMap(rr, cc);
                theta_start_deg(idx) = thetaEdges(cc);
                theta_end_deg(idx) = thetaEdges(cc + 1);
                r_inner(idx) = rEdges(rr);
                r_outer(idx) = rEdges(rr + 1);
                idx = idx + 1;
            end
        end
    end

    T = table(unit_index, unit_id, row, col, rf_value, ...
        occupancy_time_sec, theta_start_deg, theta_end_deg, r_inner, ...
        r_outer, rf_rows, rf_cols, total_deg_col, inner_blank_rows, ...
        polar_plot_radius);
    T.response_units = repmat("spike_count_in_analysis_window", rowCount, 1);
    if isVerticalBar
        countByBin = permute(reshape(barCoverage.nativePixelTrialCount, ...
            barCoverage.pixelsPerBin, []), [2 1]);
        exposureByBin = permute(reshape(barCoverage.nativePixelExposureSec, ...
            barCoverage.pixelsPerBin, []), [2 1]);
        T.event_definition = repmat( ...
            "trial_onset_for_each_covering_" + barColor + "_bar", rowCount, 1);
        T.bar_width_handling = repmat("pooled", rowCount, 1);
        T.pooled_bar_widths_deg = repmat( ...
            string(mat2str(barCoverage.barWidthsDeg)), rowCount, 1);
        T.pooled_bar_widths_pix = repmat( ...
            string(mat2str(barCoverage.barWidthsPix)), rowCount, 1);
        T.pooled_bar_width_trial_counts = repmat( ...
            string(mat2str(barCoverage.barWidthTrialCount)), rowCount, 1);
        T.bar_bin_width_deg = barBinWidthDeg * ones(rowCount, 1);
        T.bar_height_pix = screenHeightPix * ones(rowCount, 1);
        T.screen_width_pix = screenWidthPix * ones(rowCount, 1);
        T.screen_height_pix = screenHeightPix * ones(rowCount, 1);
        T.mean_native_pixel_trial_count = repmat( ...
            mean(countByBin, 2), unitNum * rfRows, 1);
        T.min_native_pixel_trial_count = repmat( ...
            min(countByBin, [], 2), unitNum * rfRows, 1);
        T.max_native_pixel_trial_count = repmat( ...
            max(countByBin, [], 2), unitNum * rfRows, 1);
        T.mean_native_pixel_exposure_sec = repmat( ...
            mean(exposureByBin, 2), unitNum * rfRows, 1);
    end
%%%%%egocentric%%%%%
    if exist(csvFile, 'file') == 2
        delete(csvFile);
    end
%%%%%egocentric%%%%%
    writetable(T, csvFile);
end
