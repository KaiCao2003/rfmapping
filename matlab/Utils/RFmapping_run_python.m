function RFmapping_run_python(rfmapPath, probe, params)
    commandParts = { ...
        ShellQuote(params.rfPythonExecutable), ...
        ShellQuote(params.rfPythonScript), ...
        ShellQuote(rfmapPath), ...
        '--probe', ShellQuote(probe), ...
        '--time-range', sprintf('%.17g', params.rfTimeRange(1)), ...
        sprintf('%.17g', params.rfTimeRange(2)), ...
        '--max-zero-bins', sprintf('%d', params.maxZeroBins), ...
        '--cluster-forming-z-2d', sprintf('%.17g', params.clusterFormingZ2d), ...
        '--cluster-forming-z-1d', sprintf('%.17g', params.clusterFormingZ1d), ...
        '--drop-bins', sprintf('%d', params.dropBins)};
    if params.rfCollapseFrom2d
        commandParts{end + 1} = '--collapse-from-2d';
    end
    if ~params.rfWrapX
        commandParts{end + 1} = '--no-wrap-x';
    end

    [rfDirectory, rfName] = fileparts(rfmapPath);
    % Paired notebooks use the regular ON source; other maps need distinct CSV targets.
    if ~isempty(regexp(rfName, ['^regular_unitsSpikeCounts_', params.date, '_[0-9]+$'], 'once'))
        [~, mouse] = fileparts(regexprep(params.base_dir, '[\\/]+$', ''));
        dataDirectory = rfDirectory;
        for level = 1:4
            dataDirectory = fileparts(dataDirectory);
        end
        commandParts = [commandParts, { ...
            '--unit-prefix', ShellQuote([mouse, ':', params.date, ':', probe]), ...
            '--comparison-output-dir', ShellQuote(fullfile(dataDirectory, 'tc_comparison'))}];
    end

    [status, output] = system([strjoin(commandParts, ' '), ' 2>&1']);
    if status ~= 0
        error('RFmapping:PythonDetectionFailed', ...
            'RF detection failed for %s (exit %d):\n%s', rfmapPath, status, output);
    end
    fprintf('%s', output);
end

function quoted = ShellQuote(value)
    % POSIX single quotes preserve paths containing spaces, quotes, and shell syntax.
    quote = char(39);
    quoted = [quote, strrep(char(value), quote, [quote, '"', quote, '"', quote]), quote];
end
