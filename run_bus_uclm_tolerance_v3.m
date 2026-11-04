function run_bus_uclm_tolerance_v3()
% Frozen-mask box perturbation study, MATLAB R2025a.
% Edit only the CONFIG block, then run this function. Requires Image
% Processing, Deep Learning, and Statistics and Machine Learning Toolboxes.
% No retraining and no selection of a margin on BUS-UCLM are performed.

%% CONFIG: confirmed dataset location; no path editing required
root = 'C:\2026\publications\Medical Imaging and Physicals\BUS-UCLM';
cfg.manifest = fullfile(root,'bus_uclm_manifest.csv');
cfg.modelFiles = {}; % selected through a MATLAB file dialog below
cfg.outputDir = fullfile(root,'localisation_tolerance_results');
cfg.malignantClass = 'malignant';
cfg.inputSize = [224 224];
cfg.contextFactor = 1.2;
cfg.bootstrapReplicates = 1000;
cfg.bootstrapSeed = 9042026;
cfg.expectedOracleAUC = 0.832;
cfg.oracleTolerance = 0.03;
cfg.fullImageScores = ''; % optional matched full-resize scores CSV

if ~isfile(cfg.manifest)
    buildManifest(root,cfg.manifest);
end
[f,modelFolder] = uigetfile({'*.mat','Trained ROI 1.2x network MAT files'}, ...
    'Select the saved ROI 1.2x network models (all available seeds)', ...
    'MultiSelect','on');
assert(~isequal(f,0),'No model was selected. Saved ROI 1.2x networks are required.');
if ischar(f), f={f}; end
cfg.modelFiles = cellfun(@(x)fullfile(modelFolder,x),f,'UniformOutput',false);

% Predeclared box errors. Shift is a fraction of the *original* lesion
% bounding-box width/height. Scale multiplies both dimensions about centre.
shiftFractions = [0.05 0.10 0.20];
scaleFactors = [0.80 0.90 1.10 1.20];
directions = [1 0; -1 0; 0 1; 0 -1; 1 1; 1 -1; -1 1; -1 -1];

assert(isfile(cfg.manifest), 'Manifest file does not exist: %s',cfg.manifest);
assert(~isempty(cfg.modelFiles) && all(cellfun(@isfile,cfg.modelFiles)), ...
    'Set cfg.modelFiles to saved trained ROI 1.2x networks.');
if ~isfolder(cfg.outputDir), mkdir(cfg.outputDir); end
T = readtable(cfg.manifest,'TextType','string');
need = ["case_id","image_path","mask_path","label"];
assert(all(ismember(need,string(T.Properties.VariableNames))), ...
    'Manifest columns must be case_id,image_path,mask_path,label.');
assert(numel(unique(T.case_id))==height(T),'case_id must be unique.');
y = double(strcmpi(strtrim(T.label),"malignant"));
assert(all(strcmpi(strtrim(T.label),"malignant") | ...
           strcmpi(strtrim(T.label),"benign")), 'Labels must be benign/malignant.');
assert(any(y==1) && any(y==0),'Both classes are required.');

% Build all conditions before looking at any prediction or AUC.
names = "oracle"; perturb = [0 0 1];
for f=shiftFractions
    for k=1:size(directions,1)
        names(end+1,1) = sprintf('shift_%02d_dir%02d',round(f*100),k); %#ok<AGROW>
        perturb(end+1,:) = [f*directions(k,:) 1]; %#ok<AGROW>
    end
end
for z=scaleFactors
    names(end+1,1) = sprintf('scale_%03d',round(z*100)); %#ok<AGROW>
    perturb(end+1,:) = [0 0 z]; %#ok<AGROW>
end
N=height(T); S=numel(cfg.modelFiles); K=numel(names);
assert(N==264,'Expected the manuscript BUS-UCLM cohort of 264 cases; found %d.',N);
scores=nan(N,S,K); boxes=nan(N,4); skipped=strings(0,1);

% Read every case once. Any invalid case is a hard error to prevent a
% condition-specific denominator or a changed external cohort.
images=cell(N,1);
for i=1:N
    ip=resolvePath(T.image_path(i),cfg.manifest);
    mp=resolvePath(T.mask_path(i),cfg.manifest);
    assert(isfile(ip) && isfile(mp),'Missing image or mask for %s',T.case_id(i));
    I=imread(ip); M=imread(mp);
    if size(I,3)==3, I=rgb2gray(I); end
    if size(M,3)==3, M=rgb2gray(M); end
    assert(isequal(size(I,1),size(M,1)) && isequal(size(I,2),size(M,2)), ...
        'Image/mask geometry differs for %s',T.case_id(i));
    bw=imfill(im2single(M)>0.1,'holes');
    bw=bwareaopen(bw,20);
    CC=bwconncomp(bw);
    assert(CC.NumObjects>0,'Empty cleaned mask for %s',T.case_id(i));
    [~,j]=max(cellfun(@numel,CC.PixelIdxList));
    [rr,cc]=ind2sub(size(bw),CC.PixelIdxList{j});
    xmin=min(cc);xmax=max(cc);ymin=min(rr);ymax=max(rr);
    boxes(i,:)=[(xmin+xmax)/2,(ymin+ymax)/2,xmax-xmin+1,ymax-ymin+1];
    images{i}=I;
end

for s=1:S
    A=load(cfg.modelFiles{s});
    fields=fieldnames(A); found={};
    for z=1:numel(fields)
        candidate=A.(fields{z});
        if isa(candidate,'SeriesNetwork') || isa(candidate,'DAGNetwork')
            found{end+1}=fields{z}; %#ok<AGROW>
        end
    end
    assert(numel(found)==1, ...
      'Expected exactly one SeriesNetwork/DAGNetwork in %s; found %d.', ...
      cfg.modelFiles{s},numel(found));
    net=A.(found{1});
    cls=string(net.Layers(end).Classes);
    pos=find(strcmpi(cls,cfg.malignantClass));
    assert(isscalar(pos),'Cannot identify malignant score column in model %d.',s);
    fprintf('Model %d/%d: %s\n',s,S,cfg.modelFiles{s});
    for k=1:K
        Xbatch=zeros(cfg.inputSize(1),cfg.inputSize(2),3,N,'single');
        for i=1:N
            b=boxes(i,:);
            b(1)=b(1)+perturb(k,1)*b(3);
            b(2)=b(2)+perturb(k,2)*b(4);
            b(3:4)=b(3:4)*perturb(k,3);
            Xbatch(:,:,:,i)=makeInput(images{i},b,cfg.contextFactor,cfg.inputSize);
        end
        P=predict(net,Xbatch,'MiniBatchSize',32,'ExecutionEnvironment','auto');
        assert(size(P,1)==N && size(P,2)==numel(cls) && all(isfinite(P(:))), ...
            'Invalid network scores for model %d, condition %s.',s,names(k));
        scores(:,s,k)=double(P(:,pos));
        fprintf('  %-20s AUC %.4f\n',names(k),auc(y,scores(:,s,k)));
    end
end

oracleAUC=mean(arrayfun(@(s)auc(y,scores(:,s,1)),1:S));
if isfinite(cfg.expectedOracleAUC)
    assert(abs(oracleAUC-cfg.expectedOracleAUC)<=cfg.oracleTolerance, ...
      ['Oracle AUC %.4f does not reproduce the manuscript %.4f (+/- %.3f). ' ...
       'Check exact preprocessing, case cohort, class order and saved networks.'], ...
       oracleAUC,cfg.expectedOracleAUC,cfg.oracleTolerance);
end

% Fixed-stratum paired bootstrap: resample benign and malignant case indices
% once per replicate; reuse those indices for every condition and model seed.
rng(cfg.bootstrapSeed,'twister');
ix0=find(y==0);ix1=find(y==1);
bootIdx=zeros(N,cfg.bootstrapReplicates);
for b=1:cfg.bootstrapReplicates
    bootIdx(:,b)=[ix0(randi(numel(ix0),numel(ix0),1)); ...
                  ix1(randi(numel(ix1),numel(ix1),1))];
end
aucBySeed=zeros(S,K); deltaBySeed=zeros(S,K);
for s=1:S
    for k=1:K
        aucBySeed(s,k)=auc(y,scores(:,s,k));
        deltaBySeed(s,k)=aucBySeed(s,k)-aucBySeed(s,1);
    end
end
R=table(names,mean(aucBySeed,1)',std(aucBySeed,0,1)', ...
    mean(deltaBySeed,1)','VariableNames', ...
    {'condition','auc_mean','auc_seed_sd','delta_vs_oracle'});
lo=nan(K,1);hi=nan(K,1);
for k=1:K
    D=zeros(cfg.bootstrapReplicates,1);
    for b=1:cfg.bootstrapReplicates
        ix=bootIdx(:,b); v=zeros(S,1);
        for s=1:S
            v(s)=auc(y(ix),scores(ix,s,k))-auc(y(ix),scores(ix,s,1));
        end
        D(b)=mean(v);
    end
    ci=prctile(D,[2.5,97.5]);lo(k)=ci(1);hi(k)=ci(2);
end
R.delta_ci_low=lo;R.delta_ci_high=hi;
writetable(R,fullfile(cfg.outputDir,'condition_summary.csv'));

% Optional matched full-image comparison. Scores must correspond to the
% identical case IDs and to seed numbers 1..S in cfg.modelFiles order.
if strlength(string(cfg.fullImageScores))>0
    F=readtable(cfg.fullImageScores,'TextType','string');
    assert(all(ismember(["case_id","seed","score"], ...
        string(F.Properties.VariableNames))),'Full-image CSV columns: case_id,seed,score.');
    base=nan(N,S);
    for s=1:S
        Q=F(F.seed==s,:);
        assert(height(Q)==N && numel(unique(Q.case_id))==N,'Full-image seed %d is incomplete.',s);
        [ok,loc]=ismember(T.case_id,Q.case_id);
        assert(all(ok),'Full-image case IDs differ at seed %d.',s);
        base(:,s)=Q.score(loc);
    end
    out=zeros(K,3);
    for k=1:K
        d=zeros(cfg.bootstrapReplicates,1);
        for b=1:cfg.bootstrapReplicates
            ix=bootIdx(:,b);v=zeros(S,1);
            for s=1:S
                v(s)=auc(y(ix),scores(ix,s,k))-auc(y(ix),base(ix,s));
            end
            d(b)=mean(v);
        end
        out(k,1)=mean(aucBySeed(:,k)-arrayfun(@(s)auc(y,base(:,s)),(1:S)'));
        out(k,2:3)=prctile(d,[2.5 97.5]);
    end
    R.delta_vs_full=out(:,1);R.full_ci_low=out(:,2);R.full_ci_high=out(:,3);
    writetable(R,fullfile(cfg.outputDir,'condition_summary.csv'));
end

% At each shift magnitude, the weakest of the eight prespecified directions
% defines a conservative sensitivity summary (not a significance test).
W=table(shiftFractions(:),nan(numel(shiftFractions),1), ...
    strings(numel(shiftFractions),1),'VariableNames', ...
    {'shift_fraction','worst_auc','worst_direction'});
for j=1:numel(shiftFractions)
    rows=2+(j-1)*size(directions,1):1+j*size(directions,1);
    [W.worst_auc(j),t]=min(R.auc_mean(rows));
    W.worst_direction(j)=R.condition(rows(t));
end
if ismember('full_ci_low',R.Properties.VariableNames)
    W.minimum_delta_ci_low=nan(height(W),1);
    for j=1:height(W)
        rows=2+(j-1)*size(directions,1):1+j*size(directions,1);
        W.minimum_delta_ci_low(j)=min(R.full_ci_low(rows));
    end
    W.all_directions_lower_ci_above_zero=W.minimum_delta_ci_low>0;
end
writetable(W,fullfile(cfg.outputDir,'shift_tolerance_summary.csv'));

% Case-level prediction archive enables independent recalculation.
caseID=repmat(T.case_id,S*K,1);
label=repmat(T.label,S*K,1);
seed=repelem((1:S)',N*K); condition=strings(N*S*K,1);score=zeros(N*S*K,1);
q=0;
for s=1:S
    for k=1:K
        for i=1:N
            q=q+1;caseID(q)=T.case_id(i);label(q)=T.label(i);
            seed(q)=s;condition(q)=names(k);score(q)=scores(i,s,k);
        end
    end
end
writetable(table(caseID,label,seed,condition,score), ...
    fullfile(cfg.outputDir,'case_predictions.csv'));
save(fullfile(cfg.outputDir,'locked_analysis.mat'),'cfg','names','perturb', ...
    'boxes','scores','y','T','R','aucBySeed','deltaBySeed','skipped','-v7.3');
disp(R);
fprintf('Saved results to %s\n',cfg.outputDir);
end

function p=resolvePath(s,manifest)
p=char(s);
if isfile(p),return;end
p=fullfile(fileparts(manifest),p);
end

function X=makeInput(I,b,contextFactor,outSize)
% Replace this function with the *exact* crop/normalisation routine used
% to train the saved ROI model if its implementation differs.
cx=b(1);cy=b(2);w=contextFactor*b(3);h=contextFactor*b(4);
x1=max(1,floor(cx-w/2));x2=min(size(I,2),ceil(cx+w/2));
y1=max(1,floor(cy-h/2));y2=min(size(I,1),ceil(cy+h/2));
assert(x2>=x1 && y2>=y1,'Invalid crop after perturbation.');
C=single(I(y1:y2,x1:x2));
C=(C-min(C(:)))/(max(C(:))-min(C(:))+eps('single'));
side=max(size(C,1),size(C,2));
P=zeros(side,side,'single');
row=floor((side-size(C,1))/2)+1;col=floor((side-size(C,2))/2)+1;
P(row:row+size(C,1)-1,col:col+size(C,2)-1)=C;
X=repmat(imresize(P,outSize),1,1,3);
end

function a=auc(y,score)
n1=nnz(y==1);n0=nnz(y==0);
r=tiedrank(score);
a=(sum(r(y==1))-n1*(n1+1)/2)/(n1*n0);
end

function buildManifest(root,outputFile)
% Build the 264 lesion-case manifest from the BUS-UCLM RGB annotations.
% Uses the directory confirmed by the user; no folder dialogs or old helper.
imageDir = fullfile(root,'images');
maskDir = fullfile(root,'masks');
assert(isfolder(imageDir) && isfolder(maskDir), ...
    'Expected images and masks folders beneath %s',root);

masks = dir(fullfile(maskDir,'*.png'));
assert(~isempty(masks),'No PNG masks found in %s',maskDir);
case_id = strings(0,1);
image_path = strings(0,1);
mask_path = strings(0,1);
label = strings(0,1);
normalCount = 0;

for j=1:numel(masks)
    maskFile = fullfile(maskDir,masks(j).name);
    imageFile = fullfile(imageDir,masks(j).name);
    assert(isfile(imageFile),'Missing matching image: %s',imageFile);
    M = imread(maskFile);
    assert(size(M,3)==3,'Expected an RGB mask: %s',maskFile);
    R=double(M(:,:,1)); G=double(M(:,:,2)); B=double(M(:,:,3));
    % Black/neutral masks represent normal images. Count only pixels whose
    % red or green channel dominates both alternatives.
    redCount = nnz(R>20 & R>G & R>B);
    greenCount = nnz(G>20 & G>R & G>B);
    if redCount==0 && greenCount==0
        normalCount=normalCount+1;
        continue;
    end
    assert(xor(redCount>0,greenCount>0), ...
        'Mask has both red and green lesion pixels: %s',maskFile);
    [~,stem]=fileparts(masks(j).name);
    case_id(end+1,1)=string(stem); %#ok<AGROW>
    image_path(end+1,1)=string(imageFile); %#ok<AGROW>
    mask_path(end+1,1)=string(maskFile); %#ok<AGROW>
    if redCount>0
        label(end+1,1)="malignant"; %#ok<AGROW>
    else
        label(end+1,1)="benign"; %#ok<AGROW>
    end
end

fprintf('Masks: %d; normal: %d; benign: %d; malignant: %d\n', ...
    numel(masks),normalCount,nnz(label=="benign"),nnz(label=="malignant"));
assert(numel(case_id)==264 && nnz(label=="benign")==174 && ...
    nnz(label=="malignant")==90, ...
    'Counts differ from manuscript. Manifest was not written.');
assert(numel(unique(case_id))==264,'Duplicate case IDs.');
T=sortrows(table(case_id,image_path,mask_path,label),'case_id');
writetable(T,outputFile);
fprintf('Manifest written: %s\n',outputFile);
end
