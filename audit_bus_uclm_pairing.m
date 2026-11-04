function audit_bus_uclm_pairing()
% Run in MATLAB R2025a: audit_bus_uclm_pairing
% Audits saved score/label consistency and locates the original source code.
% A static search cannot by itself prove case identity across modes.

searchRoot = 'C:\2026\publications';
projectRoot = fullfile(searchRoot,'Medical Imaging and Physicals');
matFile = fullfile(projectRoot, ...
    'outputs_ultrasound_revised_publication_roi1p2','main_results_revised.mat');
reportFile = fullfile(projectRoot,'bus_uclm_pairing_audit.txt');
assert(isfile(matFile),'Missing manuscript results: %s',matFile);
S=load(matFile,'harmResults','busuclmRecords');
assert(isfield(S,'harmResults') && isfield(S,'busuclmRecords'), ...
    'Expected harmResults and busuclmRecords in the MAT file.');

fid=fopen(reportFile,'w');
assert(fid>0,'Cannot open report: %s',reportFile);
cleanup=onCleanup(@() fclose(fid)); %#ok<NASGU>
writeBoth(fid,'BUS-UCLM CASE ORDER AUDIT\n');
writeBoth(fid,'Results file: %s\n\n',matFile);

T=S.busuclmRecords;
writeBoth(fid,'Record table: %d rows\n',height(T));
writeBoth(fid,'Record columns: %s\n',strjoin(string(T.Properties.VariableNames),', '));
assert(height(T)==264,'Expected 264 BUS-UCLM records.');
ids=identifierColumn(T);
if strlength(ids)>0
    v=string(T.(ids));
    writeBoth(fid,'Candidate identifier column: %s; unique=%d\n',ids,numel(unique(v)));
    writeBoth(fid,'First five IDs: %s\n',strjoin(v(1:min(5,numel(v))),', '));
else
    writeBoth(fid,'No obvious identifier column was found in busuclmRecords.\n');
end

modes=fieldnames(S.harmResults);
ref=string(S.harmResults.roi_1p2.seed(1).BUSUCLM.YTrue);
assert(numel(ref)==264,'Expected 264 reference labels.');
assert(nnz(strcmpi(ref,'benign'))==174 && ...
       nnz(strcmpi(ref,'malignant'))==90,'Unexpected label counts.');
allLabelsMatch=true;allAucMatch=true;
writeBoth(fid,'\nSAVED RESULT CHECKS\n');
for m=1:numel(modes)
    Z=S.harmResults.(modes{m}).seed;
    writeBoth(fid,'%s: %d seeds\n',modes{m},numel(Z));
    if numel(Z)~=10,allLabelsMatch=false;end
    for s=1:numel(Z)
        x=Z(s).BUSUCLM;
        y=string(x.YTrue);
        score=double(x.scorePos(:));
        labelsOK=numel(y)==264 && isequal(y,ref);
        scoresOK=numel(score)==264 && all(isfinite(score));
        if labelsOK && scoresOK
            auc=rankAuc(strcmpi(y,'malignant'),score);
            aucOK=abs(auc-double(x.auc))<1e-4;
        else
            auc=NaN;aucOK=false;
        end
        allLabelsMatch=allLabelsMatch && labelsOK && scoresOK;
        allAucMatch=allAucMatch && aucOK;
        if ~labelsOK || ~scoresOK || ~aucOK
            writeBoth(fid,'  seed %d: labels=%d scores=%d AUC=%d recomputed=%.6f\n', ...
                s,labelsOK,scoresOK,aucOK,auc);
        end
    end
end
writeBoth(fid,'All modes/seeds have identical label sequences: %d\n',allLabelsMatch);
writeBoth(fid,'All saved AUCs reproduce from scorePos: %d\n',allAucMatch);

writeBoth(fid,'\nSOURCE SEARCH\n');
files=dir(fullfile(searchRoot,'**','*.m'));
hits=0;
for k=1:numel(files)
    path=fullfile(files(k).folder,files(k).name);
    if contains(lower(path),'audit_bus_uclm_pairing'),continue;end
    try
        raw=fileread(path);
    catch
        continue;
    end
    if ~contains(raw,'harmResults') || ~contains(raw,'busuclmRecords')
        continue;
    end
    hits=hits+1;
    writeBoth(fid,'\nCANDIDATE %d: %s\n',hits,path);
    lines=regexp(raw,'\r\n|\n|\r','split');
    key=false(numel(lines),1);
    for j=1:numel(lines)
        key(j)=contains(lines{j},'harmResults') || ...
            contains(lines{j},'busuclmRecords') || ...
            contains(lines{j},'scorePos');
    end
    ix=find(key);
    if numel(ix)>60
        writeBoth(fid,'Showing first 60 of %d matched lines.\n',numel(ix));
        ix=ix(1:60);
    end
    show=false(numel(lines),1);
    for j=ix(:)'
        show(max(1,j-8):min(numel(lines),j+8))=true;
    end
    displayed=find(show);
    for j=displayed(:)'
        writeBoth(fid,'%5d | %s\n',j,lines{j});
    end
end
writeBoth(fid,'\nMatching original source files: %d\n',hits);
writeBoth(fid,['AUDIT STATUS: Saved labels and AUCs can be checked automatically; ' ...
    'case identity is NOT PROVEN by equal labels. Review the matched source ' ...
    'blocks above to confirm that the same unchanged busuclmRecords order ' ...
    'was used to form scorePos for every mode and seed.\n']);
writeBoth(fid,'Report: %s\n',reportFile);
end

function id=identifierColumn(T)
id="";
n=string(T.Properties.VariableNames);
preferred=["case_id","caseid","image_path","imagepath", ...
    "filename","file_name","imagefile","filepath","name"];
for x=preferred
    j=find(strcmpi(n,x),1);
    if ~isempty(j),id=n(j);return;end
end
end

function a=rankAuc(y,s)
n1=nnz(y);n0=nnz(~y);
r=tiedrank(s);
a=(sum(r(y))-n1*(n1+1)/2)/(n1*n0);
end

function writeBoth(fid,varargin)
fprintf(fid,varargin{:});
fprintf(varargin{:});
end