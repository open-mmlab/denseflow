
function [] = extractOpticalFlow_gpu(index, device_id, type)
% path1 = '/nfs/lmwang/lmwang/Data/UCF101/ucf101_org/';
% if type ==0
%     path2 = '/media/sdb/lmwang/data/UCF101/ucf101_flow_img_farn_gpu_step_2/';
% elseif type ==1
%     path2 = '/media/sdb/lmwang/data/UCF101/ucf101_flow_img_tvl1_gpu_step_2/';
% else
%     path2 = '/media/sdb/lmwang/data/UCF101/ucf101_flow_img_brox_gpu_step_2/';
% end


path1 = '/nfs/lmwang/lmwang/Data/HMDB51/hmdb51_org/';
if type ==0
    path2 = '/media/sdb/lmwang/data/HMDB51/hmdb51_flow_img_farn_gpu/';
elseif type ==1
    path2 = '/media/sdb/lmwang/data/HMDB51/hmdb51_flow_img_tvl1_gpu/';
else
    path2 = '/media/sdb/lmwang/data/HMDB51/hmdb51_flow_img_brox_gpu/';
end
folderlist = dir(path1);
foldername = {folderlist(:).name};
foldername = setdiff(foldername,{'.','..'});

for i = index
    if ~exist([path2,foldername{i}],'dir')
        mkdir([path2,foldername{i}]);
    end
    filelist = dir([path1,foldername{i},'/*.avi']);

    for j = 1:length(filelist)
        if ~exist([path2,foldername{i},'/',filelist(j).name(1:end-4)],'dir')
            mkdir([path2,foldername{i},'/',filelist(j).name(1:end-4)]);
        end
        file1 = [path1,foldername{i},'/',filelist(j).name];
        file2 = [path2,foldername{i},'/',filelist(j).name(1:end-4),'/','flow_x'];
        file3 = [path2,foldername{i},'/',filelist(j).name(1:end-4),'/','flow_y'];
		file4 = [path2,foldername{i},'/',filelist(j).name(1:end-4),'/','flow_i'];
        cmd = sprintf('./extract_gpu -f %s -x %s -y %s -i %s -b 20 -t %d -d %d -s %d',...
            file1,file2,file3,file4,type,device_id,1);
        system(cmd);
	end
	i
end
end
