clear
close all
imtool close all

%　★:実行前確認


%% 処理するjsonの選択
cd('D:\20241016_mocap_kinect同時計測\20241016_mocap_kinect同時計測\json');      % ←←←←←←←←jsonファイルがあるフォルダのパスを記入★
%パスの取得
selpath = uigetdir; %フォルダの選択 　        
[filepath,currentfoldername,ext] = fileparts(selpath); %現在のフォルダ名を取得
cd (selpath); %フォルダの移動

SubFolder = dir(selpath);
SubFolder = SubFolder(~ismember({SubFolder.name},{'.','..','.DS_Store'}));
SubFolderName = {SubFolder.name};

title = erase(currentfoldername,"use_");%excelの名前を編集



%% 座標データを取得

    files = dir('**/*.json'); %ファイル読み込み
    count = length(files); %jsonファイル数カウントfid = fopen(fname);


    %jsonから座標データの取得
    %カラの配列を作成
    R_hip_x  = zeros(count,1);
    R_hip_y  = zeros(count,1);
    R_knee_x  = zeros(count,1);
    R_knee_y  = zeros(count,1);
    R_ankle_x  = zeros(count,1);
    R_ankle_y  = zeros(count,1);
    R_bigtoe_x = zeros(count,1);
    R_bigtoe_y = zeros(count,1);
    R_heel_x = zeros(count,1);
    R_heel_y = zeros(count,1);

    L_hip_x  = zeros(count,1);
    L_hip_y  = zeros(count,1);
    L_knee_x  = zeros(count,1);
    L_knee_y  = zeros(count,1);
    L_ankle_x  = zeros(count,1);
    L_ankle_y  = zeros(count,1);
    L_bigtoe_x = zeros(count,1);
    L_bigtoe_y = zeros(count,1);
    L_heel_x = zeros(count,1);
    L_heel_y = zeros(count,1);



    %複数のjsonを順番に読み込む
    for i = 1:count
    fname = files(i).name;
    fid = fopen(fname);
    raw = fread(fid,inf);
    str = char(raw');
    fclose(fid);
    val = jsondecode(str);

        if isempty(val.people) == 0  %人物を検知できているかどうか　出来てるなら０になる
            %座標の取得
            r_hip_x  = val.people(1).pose_keypoints_2d(28,1);
            r_hip_y  = val.people(1).pose_keypoints_2d(29,1);
            r_knee_x  = val.people(1).pose_keypoints_2d(31,1);
            r_knee_y  = val.people(1).pose_keypoints_2d(32,1);
            r_ankle_x  = val.people(1).pose_keypoints_2d(34,1);
            r_ankle_y  = val.people(1).pose_keypoints_2d(35,1);
            r_bigtoe_x  = val.people(1).pose_keypoints_2d(67,1);
            r_bigtoe_y  = val.people(1).pose_keypoints_2d(68,1);
            r_heel_x  = val.people(1).pose_keypoints_2d(73,1);
            r_heel_y  = val.people(1).pose_keypoints_2d(74,1);
            l_hip_x  = val.people(1).pose_keypoints_2d(37,1);
            l_hip_y  = val.people(1).pose_keypoints_2d(38,1);
            l_knee_x  = val.people(1).pose_keypoints_2d(40,1);
            l_knee_y  = val.people(1).pose_keypoints_2d(41,1);
            l_ankle_x  = val.people(1).pose_keypoints_2d(43,1);
            l_ankle_y  = val.people(1).pose_keypoints_2d(44,1);
            l_bigtoe_x  = val.people(1).pose_keypoints_2d(58,1);
            l_bigtoe_y  = val.people(1).pose_keypoints_2d(59,1);
            l_heel_x  = val.people(1).pose_keypoints_2d(64,1);
            l_heel_y  = val.people(1).pose_keypoints_2d(65,1);




            if  (r_hip_x == 0) || (r_hip_y == 0) || (r_knee_x == 0) || (r_knee_y == 0) || (r_ankle_x == 0) || (r_ankle_y == 0) ||(r_bigtoe_x == 0)|| (r_bigtoe_y == 0) || (r_heel_x == 0) || (r_heel_y == 0)
                r_hip_x=0;
                r_hip_y=0;
                r_knee_x=0;
                r_knee_y=0;
                r_ankle_x=0;
                r_ankle_y=0;
                r_bigtoe_x=0;
                r_bigtoe_y=0;
                r_heel_x=0;
                r_heel_y=0;


            elseif (l_hip_x == 0) || (l_hip_y == 0) || (l_knee_x == 0) || (l_knee_y == 0) || (l_ankle_x == 0) || (l_ankle_y == 0) ||(l_bigtoe_x == 0)|| (l_bigtoe_y == 0) || (l_heel_x == 0) || (l_heel_y == 0)
                l_hip_x=0;
                l_hip_y=0;
                l_knee_x=0;
                l_knee_y=0;
                l_ankle_x=0;
                l_ankle_y=0;
                r_bigtoe_x=0;
                l_bigtoe_y=0;
                l_heel_x=0;
                l_heel_y=0;



            end
            
            
            R_hip_x(i,1)  = r_hip_x;
            R_hip_y(i,1)  = r_hip_y;
            R_knee_x(i,1)  = r_knee_x;
            R_knee_y(i,1) = r_knee_y;
            R_ankle_x(i,1)  = r_ankle_x;
            R_ankle_y(i,1)  = r_ankle_y;
            R_bigtoe_x(i,1) = r_bigtoe_x;
            R_bigtoe_y(i,1) = r_bigtoe_y;
            R_heel_x(i,1) = r_heel_x;
            R_heel_y(i,1) = r_heel_y;


            L_hip_x(i,1)  = l_hip_x;
            L_hip_y(i,1)  = l_hip_y;
            L_knee_x(i,1)  = l_knee_x;
            L_knee_y(i,1)  = l_knee_y;
            L_ankle_x(i,1)  = l_ankle_x;
            L_ankle_y(i,1)  = l_ankle_y;
            L_bigtoe_x(i,1) = l_bigtoe_x;
            L_bigtoe_y(i,1) = l_bigtoe_y;
            L_heel_x(i,1) = l_heel_x;
            L_heel_y(i,1) = l_heel_y;

        end

    end

    %% excelに書き込み
   
  cd ('D:\20241016_mocap_kinect同時計測\20241016_mocap_kinect同時計測\json_excel\480x480');   % ←←←←←←←←保存先のパスを記入★


    filename = [title '.xlsx'];%ファイル名をデータ名にする
    writematrix(R_hip_x,filename,'Sheet',1,'Range','A1');
    writematrix(R_hip_y,filename,'Sheet',1,'Range','B1');
    writematrix(R_knee_x,filename,'Sheet',1,'Range','C1');
    writematrix(R_knee_y,filename,'Sheet',1,'Range','D1');
    writematrix(R_ankle_x,filename,'Sheet',1,'Range','E1');
    writematrix(R_ankle_y,filename,'Sheet',1,'Range','F1');
    writematrix(R_bigtoe_x,filename,'Sheet',1,'Range','G1');
    writematrix(R_bigtoe_y,filename,'Sheet',1,'Range','H1');
    writematrix(R_heel_x,filename,'Sheet',1,'Range','I1');
    writematrix(R_heel_y,filename,'Sheet',1,'Range','J1');

    
    writematrix(L_hip_x,filename,'Sheet',2,'Range','A1');
    writematrix(L_hip_y,filename,'Sheet',2,'Range','B1');
    writematrix(L_knee_x,filename,'Sheet',2,'Range','C1');
    writematrix(L_knee_y,filename,'Sheet',2,'Range','D1');
    writematrix(L_ankle_x,filename,'Sheet',2,'Range','E1');
    writematrix(L_ankle_y,filename,'Sheet',2,'Range','F1');
    writematrix(L_bigtoe_x,filename,'Sheet',2,'Range','G1');
    writematrix(L_bigtoe_y,filename,'Sheet',2,'Range','H1');
    writematrix(L_heel_x,filename,'Sheet',2,'Range','I1');
    writematrix(L_heel_y,filename,'Sheet',2,'Range','J1');


