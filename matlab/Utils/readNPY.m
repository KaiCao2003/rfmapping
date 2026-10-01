function numpyArray = readNPY(filename)
npyFileID = fopen(filename,'r');

% Read the NPY signature and skip the first 6 bytes.
fread(npyFileID,6,'*uint8')'; 

npyMajorVersion = fread(npyFileID,1,'uint8');
fread(npyFileID,1,'uint8');

if npyMajorVersion == 1
    npyHeaderLengthInBytes = fread(npyFileID,1,'uint16',0,'ieee-le');
else
    npyHeaderLengthInBytes = fread(npyFileID,1,'uint32',0,'ieee-le');
end
npyHeaderText = char(fread(npyFileID,npyHeaderLengthInBytes,'*char')');

% Pull dtype, storage order, and shape out of the Python-style header.
numpyDtypeToken = regexp(npyHeaderText, '''descr''\s*:\s*''([^'']+)''', 'tokens', 'once');
numpyFortranOrderToken = regexp(npyHeaderText, '''fortran_order''\s*:\s*(True|False)', 'tokens', 'once');
numpyArrayShapeToken = regexp(npyHeaderText, '''shape''\s*:\s*\(([^\)]*)\)', 'tokens', 'once');

numpyDtypeDescription = numpyDtypeToken{1};
numpyArrayIsStoredInFortranOrder = strcmp(numpyFortranOrderToken{1},'True');
numpyArrayShapeText = strtrim(numpyArrayShapeToken{1});
numpyArrayShape = str2double(regexp(numpyArrayShapeText, '\d+', 'match'));

% Read the flat bytes, then restore the array dimensions.
[matlabReadPrecision, machineByteOrder] = ConvertNumpyDtypeToMatlabReadSettings(numpyDtypeDescription);
numpyArray = fread(npyFileID, prod(numpyArrayShape), ['*',matlabReadPrecision], 0, machineByteOrder);
if numel(numpyArrayShape) > 1
    if numpyArrayIsStoredInFortranOrder
        numpyArray = reshape(numpyArray, numpyArrayShape);
    else
        numpyArray = reshape(numpyArray, fliplr(numpyArrayShape));
        numpyArray = permute(numpyArray, numel(numpyArrayShape):-1:1);
    end
end
fclose(npyFileID);
end

function [matlabReadPrecision, machineByteOrder] = ConvertNumpyDtypeToMatlabReadSettings(numpyDtypeDescription)
numpyByteOrderCode = numpyDtypeDescription(1);
numpyTypeCode = numpyDtypeDescription(2);
numpyBytesPerElement = str2double(numpyDtypeDescription(3:end));

if numpyByteOrderCode == '<'
    machineByteOrder = 'ieee-le';
elseif numpyByteOrderCode == '>'
    machineByteOrder = 'ieee-be';
else
    machineByteOrder = 'n';
end

switch numpyTypeCode
    case 'i'
        matlabReadPrecision = sprintf('int%d', numpyBytesPerElement*8);
    case 'u'
        matlabReadPrecision = sprintf('uint%d', numpyBytesPerElement*8);
    case 'f'
        if numpyBytesPerElement == 4
            matlabReadPrecision = 'single';
        else
            matlabReadPrecision = 'double';
        end
    case 'b'
        matlabReadPrecision = 'logical';
end
end
