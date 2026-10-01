function barCoverage = RFmapping_vertical_bar_coverage(trials, trialsTime, ...
        screenWidthPix, screenDeg, barBinWidthDeg, lum)

    % Match ReceptiveFieldMapping_bar_360degLED trial geometry exactly.

    trialCount = numel(trials);
    trialsTime = double(trialsTime(:));
    assert(numel(trialsTime) == trialCount + 1, ...
        'Vertical-bar trial timing must contain one more boundary than trials.');
    assert(all(diff(trialsTime) > 0), ...
        'Vertical-bar trial boundaries must be strictly increasing.');

    barCenterXDeg = double([trials.Square_PositionX]);
    barCenterYDeg = double([trials.Square_PositionY]);
    barWidthsDegByTrial = double([trials.Square_Size]);
    barLuminance = double([trials.Square_Luminance]);
    assert(all(barCenterYDeg == 0), ...
        'Vertical-bar trials must all have Square_PositionY equal to 0.');
    assert(all(ismember(barCenterXDeg, -174:12:174)), ...
        'Vertical-bar X positions must belong to -174:12:174 degrees.');
    assert(isequal(unique(barCenterXDeg), -174:12:174), ...
        'Vertical-bar trials must contain all positions from -174:12:174 degrees.');
    assert(all(ismember(barWidthsDegByTrial, [3 6 9 12])), ...
        'Vertical-bar widths must be 3, 6, 9, or 12 degrees.');
    assert(isequal(unique(barWidthsDegByTrial), [3 6 9 12]), ...
        'Vertical-bar trials must contain all widths 3, 6, 9, and 12 degrees.');
    assert(all(ismember(barLuminance, [0 0.5 1])), ...
        'Vertical-bar luminance must be 0, 0.5, or 1.');

    degreePerPixel = screenDeg / screenWidthPix;
    pixelsPerBin = barBinWidthDeg / degreePerPixel;
    assert(pixelsPerBin == round(pixelsPerBin), ...
        'barBinWidthDeg must contain an integer number of screen pixels.');
    pixelsPerBin = round(pixelsPerBin);
    assert(mod(screenWidthPix, pixelsPerBin) == 0, ...
        'barBinWidthDeg must divide the horizontal screen width.');

    footprint = false(trialCount, screenWidthPix);
    for trialIndex = 1:trialCount
        widthPix = ceil(barWidthsDegByTrial(trialIndex) / degreePerPixel);
        centerPix = (screenWidthPix + 1) / 2 + ...
            barCenterXDeg(trialIndex) / degreePerPixel;
        firstPix = round(centerPix - (widthPix - 1) / 2);
        lastPix = firstPix + widthPix - 1;
        assert(firstPix >= 1 && lastPix <= screenWidthPix, ...
            'Vertical bar extends beyond the recorded screen texture.');
        footprint(trialIndex, firstPix:lastPix) = true;
    end

    trialPixelCoverage = sparse(footprint & (barLuminance(:) == lum));
    trialStartTime = trialsTime(1:trialCount);
    trialDurationSec = trialsTime(2:trialCount + 1) - trialStartTime;
    binCount = screenWidthPix / pixelsPerBin;

    barCoverage.xPositionsDeg = -screenDeg / 2 + barBinWidthDeg / 2 + ...
        (0:(binCount - 1)) * barBinWidthDeg;
    barCoverage.trialPixelCoverage = trialPixelCoverage;
    barCoverage.nativePixelTrialCount = full(sum(trialPixelCoverage, 1));
    barCoverage.nativePixelExposureSec = full( ...
        trialDurationSec.' * trialPixelCoverage);
    barCoverage.binWidthDeg = barBinWidthDeg;
    barCoverage.pixelsPerBin = pixelsPerBin;
    barCoverage.trialStartTime = trialStartTime;
    barCoverage.allSessionBarWidthsDeg = unique(barWidthsDegByTrial);
    barCoverage.allSessionBarWidthsPix = ceil( ...
        barCoverage.allSessionBarWidthsDeg / degreePerPixel);
    barCoverage.barWidthsDeg = unique( ...
        barWidthsDegByTrial(barLuminance == lum));
    barCoverage.barWidthsPix = ceil( ...
        barCoverage.barWidthsDeg / degreePerPixel);
    barCoverage.barWidthTrialCount = arrayfun( ...
        @(widthDeg) nnz(barWidthsDegByTrial == widthDeg & barLuminance == lum), ...
        barCoverage.barWidthsDeg);
end
