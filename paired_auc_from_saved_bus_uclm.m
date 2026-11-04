function paired_auc_from_saved_bus_uclm()
% Paired case bootstrap for the saved 10-seed BUSI -> BUS-UCLM comparison.
% MATLAB R2025a. No retraining and no new model selection.
root = 'C:\2026\publications\Medical Imaging and Physicals';
inputFile = fullfile(root,'outputs_ultrasound_revised_publication_roi1p2', ...
    'main_results_revised.mat');
outputDir = fullfile(root,'outputs_ultrasound_revised_publication_roi1p2', ...
    'paired_case_auc');
assert(isfile(inputFile),'Missing results file: %s',inputFile);
if ~isfolder(outputDir), mkdir(outputDir); end
S = load(inputFile,'harmResults','busuclmRecords');
R = S.harmResults;
A = R.roi_1p2.seed;
B = R.full_resize.seed;
assert(numel(A)==10 && numel(B)==10,'Expected ten matched training seeds.');
n = numel(A(1).BUSUCLM.scorePos);
assert(n==264,'Expected 264 BUS-UCLM cases.');

y0 = string(A(1).BUSUCLM.YTrue);
assert(all(ismember(lower(y0),["benign","malignant"])), ...
    'Unexpected class names in YTrue.');
y = double(strcmpi(y0,'malignant'));
assert(nnz(y)==90 && nnz(~y)==174,'Unexpected BUS-UCLM class counts.');
roi = nan(n,10); full = nan(n,10);
for s=1:10
    a=A(s).BUSUCLM; b=B(s).BUSUCLM;
    assert(isequal(string(a.YTrue),y0) && isequal(string(b.YTrue),y0), ...
        'Label order differs in seed %d. Cannot pair the case rows.',s);
    roi(:,s)=double(a.scorePos(:));
    full(:,s)=double(b.scorePos(:));
    assert(all(isfinite(roi(:,s))) && all(isfinite(full(:,s))), ...
        'Nonfinite scores in seed %d.',s);
    ar=localAuc(y,roi(:,s)); af=localAuc(y,full(:,s));
    assert(abs(ar-double(a.auc))<1e-4 && abs(af-double(b.auc))<1e-4, ...
        ['AUC does not reproduce stored metrics in seed %d. Check scorePos ' ...
         'class direction before using paired estimates.'],s);
end

% The saved arrays have no case ID attached to each score. The original
% matched pipeline must have evaluated modes in the same busuclmRecords
% order. Matching YTrue vectors alone cannot prove identity alignment.
assert(height(S.busuclmRecords)==n,'Record table is not 264 cases.');
seed=(1:10)';
auc_roi=arrayfun(@(s)localAuc(y,roi(:,s)),1:10)';
auc_full=arrayfun(@(s)localAuc(y,full(:,s)),1:10)';
delta=auc_roi-auc_full;
perSeed=table(seed,auc_roi,auc_full,delta);
writetable(perSeed,fullfile(outputDir,'paired_auc_by_seed.csv'));

% Stratified paired case bootstrap. Both modes and all ten fixed seeds use
% the same resampled case indices in each replicate.
reps=5000; rng(9042026,'twister');
i0=find(y==0); i1=find(y==1);
bootDelta=nan(reps,1);
for r=1:reps
    ix=[i0(randi(numel(i0),numel(i0),1)); ...
        i1(randi(numel(i1),numel(i1),1))];
    d=zeros(10,1);
    for s=1:10
        d(s)=localAuc(y(ix),roi(ix,s))-localAuc(y(ix),full(ix,s));
    end
    bootDelta(r)=mean(d);
end
ci=prctile(bootDelta,[2.5 97.5]);
summary=table(mean(auc_roi),mean(auc_full),mean(delta),ci(1),ci(2), ...
    std(delta),reps,'VariableNames', ...
    {'roi_auc_mean','full_auc_mean','paired_delta_mean', ...
     'paired_delta_ci_low','paired_delta_ci_high','delta_seed_sd','bootstrap_reps'});
writetable(summary,fullfile(outputDir,'paired_auc_summary.csv'));
save(fullfile(outputDir,'paired_auc_audit.mat'),'summary','perSeed', ...
    'bootDelta','y','roi','full','inputFile','-v7.3');
disp(summary);
fprintf('Results: %s\n',outputDir);
fprintf(['PAIRING ASSUMPTION: scorePos row i denotes the same BUS-UCLM image ' ...
    'in every mode and seed. Verify this against the original evaluation code.\n']);
end

function a=localAuc(y,scores)
% Rank-based ROC AUC with average ranks for tied scores.
n1=nnz(y==1); n0=nnz(y==0);
r=tiedrank(scores);
a=(sum(r(y==1))-n1*(n1+1)/2)/(n1*n0);
end
