function create_bus_uclm_manifest_from_folders()
% Select separate BUS-UCLM image and RGB-mask folders, then create a manifest.
% This helper pairs filenames conservatively and checks the expected 174/90
% benign/malignant count. It never guesses a label from a folder name.
imdir=uigetdir(pwd,'Choose folder containing BUS-UCLM ultrasound images');
assert(~isequal(imdir,0),'No image folder chosen.');
% Start at the parent, rather than inside images, to avoid choosing images
% twice. The mask files are colour annotations in a separate folder.
maskdir=uigetdir(fileparts(imdir), ...
    'Choose the separate BUS-UCLM RGB ground-truth mask folder (NOT images)');
assert(~isequal(maskdir,0),'No mask folder chosen.');
assert(~strcmpi(char(imdir),char(maskdir)), ...
    'The image and mask folders are identical. Select the separate RGB annotation folder.');
[f,p]=uiputfile('bus_uclm_manifest.csv','Save the 264-case manifest');
assert(~isequal(f,0),'No destination chosen.');
I=allImages(imdir); M=allImages(maskdir);
assert(~isempty(I) && ~isempty(M),'Could not find image/mask files.');
% Quickly detect the common error of selecting a greyscale image directory
% as masks, before iterating over all pairs.
probe=imread(fullfile(M(1).folder,M(1).name));
assert(size(probe,3)==3, ...
    'Selected mask folder contains greyscale images, not RGB lesion masks: %s',maskdir);
mk=containers.Map('KeyType','char','ValueType','char');
for j=1:numel(M)
    key=normaliseStem(M(j).name);
    assert(~isKey(mk,key),'Ambiguous duplicate mask filename: %s',M(j).name);
    mk(key)=fullfile(M(j).folder,M(j).name);
end
case_id=strings(0,1);image_path=strings(0,1);
mask_path=strings(0,1);label=strings(0,1);
for j=1:numel(I)
    key=normaliseStem(I(j).name);
    if ~isKey(mk,key),continue;end
    ip=fullfile(I(j).folder,I(j).name);mp=mk(key);
    rgb=imread(mp);
    assert(size(rgb,3)==3,'Mask must be RGB: %s',mp);
    rgb=double(rgb);
    red=nnz(rgb(:,:,1)>64 & rgb(:,:,1)>1.5*rgb(:,:,2) & ...
        rgb(:,:,1)>1.5*rgb(:,:,3));
    green=nnz(rgb(:,:,2)>64 & rgb(:,:,2)>1.5*rgb(:,:,1) & ...
        rgb(:,:,2)>1.5*rgb(:,:,3));
    assert(xor(red>20,green>20), ...
        ['Mask label is ambiguous (red=%d, green=%d): %s. ' ...
         'Check that the selected folder contains RGB ground-truth masks, not ultrasound images.'], ...
         red,green,mp);
    case_id(end+1,1)=string(key); %#ok<AGROW>
    image_path(end+1,1)=string(ip); %#ok<AGROW>
    mask_path(end+1,1)=string(mp); %#ok<AGROW>
    if red>green
        label(end+1,1)="malignant"; %#ok<AGROW>
    else
        label(end+1,1)="benign"; %#ok<AGROW>
    end
end
assert(numel(case_id)==264 && nnz(label=="benign")==174 && ...
       nnz(label=="malignant")==90, ...
    'Found %d matched lesions (%d benign, %d malignant), expected 264 (174/90).', ...
    numel(case_id),nnz(label=="benign"),nnz(label=="malignant"));
assert(numel(unique(case_id))==264,'Duplicate case IDs after pairing.');
T=table(case_id,image_path,mask_path,label);
T=sortrows(T,'case_id');
writetable(T,fullfile(p,f));
fprintf('Wrote %d verified cases to %s\n',height(T),fullfile(p,f));
end

function files=allImages(root)
exts={'*.png','*.jpg','*.jpeg','*.tif','*.tiff','*.bmp'};
files=dir(fullfile(root,'**',exts{1}));
for k=2:numel(exts)
    files=[files;dir(fullfile(root,'**',exts{k}))]; %#ok<AGROW>
end
end

function key=normaliseStem(name)
[~,stem]=fileparts(name);
stem=regexprep(lower(stem),'([_ -]?(mask|masks|seg|segmentation|gt))$','');
key=char(stem);
end
