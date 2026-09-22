function SaveRfPatternPdf(RFmap, unitPool, unitNum, save_dir, total_deg, ...
        isBackgroundMoving, isRotation, isAllocentricPixelBins, isFineResolution, ...
        isVerticalBar, barBinWidthDeg, isPixelByPixelAnalysis, lumName, lumSuffix)
   % Gen PDF from here
    if isVerticalBar
        savePdfDir = sprintf('%sregular_vertical_bar_pooled_bin%gdeg', ...
            save_dir, barBinWidthDeg);
    elseif isBackgroundMoving
%%%%%egocentric%%%%%
        if isFineResolution
            savePdfDir = [save_dir, 'egocentric'];
        else
            savePdfDir = [save_dir, 'egocentric_30'];
        end
%%%%%egocentric%%%%%
    elseif isRotation
        if isFineResolution
            savePdfDir = [save_dir, 'rotation'];
        else
            savePdfDir = [save_dir, 'rotation_30'];
        end
    elseif isAllocentricPixelBins
        if isFineResolution
            savePdfDir = [save_dir, 'allocentric_pixelbins'];
        else
            savePdfDir = [save_dir, 'allocentric_pixelbins_30'];
        end
    else
        savePdfDir = [save_dir, 'regular'];
    end
    if exist(savePdfDir, 'dir') ~= 7
        mkdir(savePdfDir);
    end

    parfor k = 1:unitNum

        u = unitPool(k);
        rf = flipud(sum(RFmap{k}.(lumName).OnSet, 3));
        [rfRows, rfCols] = size(rf);
        if isVerticalBar || (isPixelByPixelAnalysis && isFineResolution)
%%%%%egocentric%%%%%
            plotHeight = 3;
            leftWidth = 16;
            innerBlankRows = 4;
            polarPadRows = 1;
            polarPlotRadius = innerBlankRows + rfRows + polarPadRows;
            polarSize = 9;
            figHeight = polarSize + 2.8;
            figWidth = leftWidth + polarSize + 4.5;
%%%%%egocentric%%%%%
        else
            plotHeight = 8;
            innerBlankRows = 4;
            binSize = plotHeight / rfRows;
            leftWidth = plotHeight * rfCols / rfRows;
            polarPadRows = 1;
            polarPlotRadius = innerBlankRows + rfRows + polarPadRows;
            polarSize = 2 * polarPlotRadius * binSize;
            figHeight = polarSize + 2.8;
            figWidth = leftWidth + polarSize + 4.5;
        end
        fig = figure('Visible','off','Units','centimeters','Position',[23.336,4.3,figWidth,figHeight]);

        leftAx = axes('Parent',fig,'Units','centimeters', ...
            'Position',[1.1,1.0 + (polarSize - plotHeight) / 2,leftWidth,plotHeight]);
        axes(leftAx);
        PlotColorMap(rf);
        leftAx.Toolbar.Visible = 'off';
        colormap gray;
        colorbar;
        if isVerticalBar || (isPixelByPixelAnalysis && isFineResolution)
%%%%%egocentric%%%%%
            axis tight;
%%%%%egocentric%%%%%
        else
            axis image;
        end
        if isVerticalBar
            title(['Bar ', lumName, ' trial-onset RFmap (spike count)']);
        else
            title([lumName, ' onset RFmap']);
        end
        colorLimits = get(leftAx,'CLim');

        polarAx = axes('Parent',fig,'Units','centimeters','Position',[leftWidth+3.2,1.0,polarSize,polarSize]);
        polarAx.Toolbar.Visible = 'off';
        hold(polarAx,'on');
        thetaEdges = deg2rad(linspace(90 + total_deg/2, 90 - total_deg/2, rfCols + 1));
        rEdges = innerBlankRows:(innerBlankRows + rfRows);
        outerRadius = innerBlankRows + rfRows;
        nArc = 16;
        for row = 1:rfRows
            for col = 1:rfCols
                thetaCell = linspace(thetaEdges(col), thetaEdges(col + 1), nArc);
                xPatch = [rEdges(row + 1) .* cos(thetaCell), rEdges(row) .* cos(fliplr(thetaCell))];
                yPatch = [rEdges(row + 1) .* sin(thetaCell), rEdges(row) .* sin(fliplr(thetaCell))];
                patch(polarAx,'XData',xPatch,'YData',yPatch,'CData',rf(row,col), ...
                    'FaceColor','flat','EdgeColor','none');
            end
        end
        axis(polarAx,'equal');
        axis(polarAx,'off');
        xlim(polarAx,[-polarPlotRadius polarPlotRadius]);
        ylim(polarAx,[-polarPlotRadius polarPlotRadius]);
        set(polarAx,'CLim',colorLimits);
        title(polarAx,'Polar RFmap');

        if isVerticalBar
            sgtitle(sprintf('Unit %d | pooled vertical bars | %g deg bins', ...
                u, barBinWidthDeg));
        else
            sgtitle(sprintf('Unit %d', u));
        end
        unitPdf = fullfile(savePdfDir, sprintf('%03d_unit_%d%s.pdf', k, u, lumSuffix));
        exportgraphics(fig, unitPdf, 'ContentType', 'image');
        close(fig);
        fprintf('pdf done %d out of %d\n', k, unitNum);
    end
end
