%%リセット

clear;
clc;
close all;

%　★:実行前確認
%　左脚のDoTC:Lx 左脚のDoHC:Ly 右脚のDoTC:Rx 右脚のDoHC:Ry

%% openposeから得られた座標をエクセルから取得

cd('D:\20241016_mocap_kinect同時計測\20241016_mocap_kinect同時計測\json_excel\')    %　　　　　←←←←←←←←座標データがあるエクセルのあるパスを記入★

[name_op_excel,path_op] = uigetfile('*','Select a Excel File');    %フォルダの選択
cd (path_op);  %フォルダの移動
[filepath,name,ext]=fileparts(name_op_excel); %[ファイルパス,ファイル名,拡張子]

% 座標データをexcelから取得
% 右足　
ExcelData_R = readmatrix(name_op_excel,"Sheet","Sheet1");
ankle(:,1)=ExcelData_R(:,5); %足首x
ankle(:,3)=ExcelData_R(:,6); %足首y
knee(:,1)=ExcelData_R(:,3);  %膝x
knee(:,3)=ExcelData_R(:,4);  %膝y
hip(:,1)=ExcelData_R(:,1);   %股関節x
hip(:,3)=ExcelData_R(:,2);   %股関節y
bigtoe(:,1)=ExcelData_R(:,7);%爪先x
bigtoe(:,3)=ExcelData_R(:,8);%爪先y
heel(:,1)=ExcelData_R(:,9);  %踵x
heel(:,3)=ExcelData_R(:,10); %踵y

% 左足
ExcelData_L = readmatrix(name_op_excel,"Sheet","Sheet2");
ankle(:,2)=ExcelData_L(:,5);
ankle(:,4)=ExcelData_L(:,6); 
knee(:,2)=ExcelData_L(:,3);
knee(:,4)=ExcelData_L(:,4); 
hip(:,2)=ExcelData_L(:,1);
hip(:,4)=ExcelData_L(:,2);
bigtoe(:,2)=ExcelData_L(:,7);
bigtoe(:,4)=ExcelData_L(:,8);
heel(:,2)=ExcelData_L(:,9);
heel(:,4)=ExcelData_L(:,10);

cd('D:\20241016_mocap_kinect同時計測\20241016_mocap_kinect同時計測\qualisys\')    %　　　　　←←←←←←←←3DMCの関節角度データがあるエクセルのあるパスを記入★

%% 正解角度(motion capture)を取得
[name_mc_excel,path_mc] = uigetfile('*','Select a Excel File');    %フォルダの選択
ExcelData_mc_angle = readmatrix(name_mc_excel);

frame_angle(:,1)=ExcelData_mc_angle(2:end,1); %フレーム数
hip_angle_R(:,1)=ExcelData_mc_angle(2:end,2);
knee_angle_R(:,1)=ExcelData_mc_angle(2:end,3);
ankle_angle_R(:,1)=ExcelData_mc_angle(2:end,4);
hip_angle_L(:,1)=ExcelData_mc_angle(2:end,5);
knee_angle_L(:,1)=ExcelData_mc_angle(2:end,6);
ankle_angle_L(:,1)=ExcelData_mc_angle(2:end,7);


%% 前後時間削除（取得したい時間帯のデータだけ切り取り)

timestep=30.00; % タイムステップ       ←←←←←←← 動画のフレームレートを記入★
rtimestep=1/timestep;

%エクセルから開始・終了時間を取得
beginend_excel_path = 'C:\Users\s8124\Box\相川研\個人\34期生\中崎\進捗\2025.5.8_進捗報告\コード\開始・終了時間.xlsx'; %　　←←←←←←← 取得したい時間が書いてあるexcelのパスを記入★　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　　★
[BE_filepath,BE_name,BE_ext]=fileparts(beginend_excel_path);
cd(BE_filepath)
name_beginend_excel = [BE_name BE_ext];
Excel_TimeData = readmatrix(name_beginend_excel,"Sheet","切り取り時間","Range","B2:C400");  %検索する範囲
timedata_name = readcell(name_beginend_excel,"Sheet","切り取り時間","Range","A2:A400");     %検索する範囲

%対象の撮影データ(name)の開始・終了時間を検索＆取得
for i =1:length(timedata_name)
    if strcmp(timedata_name{i},name)
        filenumber = i;
        break
    end
end

%開始と終了のインデックスを取得
front = round(Excel_TimeData(filenumber,1)*timestep);
backtime = round(length(ExcelData_R)-Excel_TimeData(filenumber,2)*timestep);
back = round(Excel_TimeData(filenumber,2)*timestep);

%開始前と終了後の座標データをカット
start_time = rtimestep*front;
cankle =ankle(front:back,:);
cknee =knee(front:back,:);
chip =hip(front:back,:);
cbigtoe =bigtoe(front:back,:);
cheel = heel(front:back,:);
ctime =transpose(start_time:rtimestep:rtimestep*(length(ankle)-backtime));


%開始前と終了後の関節角データをカット
correct_angle_frame_front=front*4; %120Hzのフレームに直す　
correct_angle_frame_back=back*4;

index_front = find(frame_angle  == correct_angle_frame_front); %120Hzのフレームの最初と最後を見つける
index_back = find(frame_angle  == correct_angle_frame_back);

adjustment_frame=1; %　  ←←←←←←モーションキャプチャとの同期のずれを補正★　      例:　1フレームずれ→adjustment_frame=2or-2　ずれなし→1

p30=4; %3DMCとopenposeの同期調整：フレーム(30fps)　
p120=p30*adjustment_frame;%←←←←←←グラフをずらすp30*a

cut_hip_cangle_R_120 =hip_angle_R(index_front+p120:index_back+p120,:);
cut_knee_cangle_R_120 =knee_angle_R(index_front+p120:index_back+p120,:);
cut_ankle_cangle_R_120 =ankle_angle_R(index_front+p120:index_back+p120,:);
cut_hip_cangle_L_120 =hip_angle_L(index_front+p120:index_back+p120,:);
cut_knee_cangle_L_120 =knee_angle_L(index_front+p120:index_back+p120,:);
cut_ankle_cangle_L_120 =ankle_angle_L(index_front+p120:index_back+p120,:);


hip_cangle_R = cut_hip_cangle_R_120(1:4:end);
knee_cangle_R = cut_knee_cangle_R_120(1:4:end);
ankle_cangle_R = cut_ankle_cangle_R_120(1:4:end);
hip_cangle_L = cut_hip_cangle_L_120(1:4:end);
knee_cangle_L = cut_knee_cangle_L_120(1:4:end);
ankle_cangle_L = cut_ankle_cangle_L_120(1:4:end);


%% openpose 補正前の位置・速度・加速度を算出 

%足首
cankle_Rx = cankle(:,1);
diff_cankle_Rx = diff(cankle_Rx);
cankle_Lx = cankle(:,2);
diff_cankle_Lx = diff(cankle_Lx);
cankle_Ry = cankle(:,3);
diff_cankle_Ry = diff(cankle_Ry);
cankle_Ly = cankle(:,4);
diff_cankle_Ly = diff(cankle_Ly);
diff2_cankle_Rx = diff(diff_cankle_Rx);
diff2_cankle_Lx = diff(diff_cankle_Lx);
diff2_cankle_Ry = diff(diff_cankle_Ry);
diff2_cankle_Ly = diff(diff_cankle_Ly);

%股関節
chip_Rx = chip(:,1);
diff_chip_Rx = diff(chip_Rx);
chip_Lx = chip(:,2);
diff_chip_Lx = diff(chip_Lx);
chip_Ry = chip(:,3);
diff_chip_Ry = diff(chip_Ry);
chip_Ly = chip(:,4);
diff_chip_Ly = diff(chip_Ly);
diff2_chip_Rx = diff(diff_chip_Rx);
diff2_chip_Lx = diff(diff_chip_Lx);
diff2_chip_Ry = diff(diff_chip_Ry);
diff2_chip_Ly = diff(diff_chip_Ly);

%膝
cknee_Rx = cknee(:,1);
diff_cknee_Rx = diff(cknee_Rx);
cknee_Lx = cknee(:,2);
diff_cknee_Lx = diff(cknee_Lx);
cknee_Ry = cknee(:,3);
diff_cknee_Ry = diff(cknee_Ry);
cknee_Ly = cknee(:,4);
diff_cknee_Ly = diff(cknee_Ly);
diff2_cknee_Rx = diff(diff_cknee_Rx);
diff2_cknee_Lx = diff(diff_cknee_Lx);
diff2_cknee_Ry = diff(diff_cknee_Ry);
diff2_cknee_Ly = diff(diff_cknee_Ly);

%つま先
cbigtoe_Rx = cbigtoe(:,1);
diff_cbigtoe_Rx = diff(cbigtoe_Rx);
cbigtoe_Lx = cbigtoe(:,2);
diff_cbigtoe_Lx = diff(cbigtoe_Lx);
cbigtoe_Ry = cbigtoe(:,3);
diff_cbigtoe_Ry = diff(cbigtoe_Ry);
cbigtoe_Ly = cbigtoe(:,4);
diff_cbigtoe_Ly = diff(cbigtoe_Ly);
diff2_cbigtoe_Rx = diff(diff_cbigtoe_Rx);
diff2_cbigtoe_Lx = diff(diff_cbigtoe_Lx);
diff2_cbigtoe_Ry = diff(diff_cbigtoe_Ry);
diff2_cbigtoe_Ly = diff(diff_cbigtoe_Ly);


%かかと
cheel_Rx = cheel(:,1);
diff_cheel_Rx = diff(cheel_Rx);
cheel_Lx = cheel(:,2);
diff_cheel_Lx = diff(cheel_Lx);
cheel_Ry = cheel(:,3);
diff_cheel_Ry = diff(cheel_Rx);
cheel_Ly = cheel(:,4);
diff_cheel_Ly = diff(cheel_Ly);
diff2_cheel_Rx = diff(diff_cheel_Rx);
diff2_cheel_Lx = diff(diff_cheel_Lx);
diff2_cheel_Ry = diff(diff_cheel_Ry);
diff2_cheel_Ly = diff(diff_cheel_Ly);


%速度と加速度のインデックスの数を調整
ctime_v = ctime;
ctime_v(end) = [];

ctime_a = ctime;
ctime_a(end) = [];
ctime_a(end) = [];


%補正前の位置と加速度を表示
%ankle_x
display_ankle_x = true ; %true:表示 false:非表示
if display_ankle_x

%速度
figure
hold on
plot(ctime,cankle_Rx,'LineWidth',0.9,Color=[0.361 0.671 0.933 1]) %右
plot(ctime,cankle_Lx,'LineWidth',0.9,Color=[0 0 1 1]) %左
set(gca, 'FontSize', 14); 
xlabel('Time [s]')
ylabel('足首のDoTC')
title('ankle_x')
hold off


%加速度
figure
hold on
plot(ctime_a,diff2_cankle_Rx,'LineWidth',0.9,Color=[0 0 0 0.9]) %右
xlabel('Time [s]')
ylabel('右足首の加速度(補正前) [m/s^2]')
set(gca, 'FontSize', 14); 
title('a')
hold off

%加速度
figure
hold on
plot(ctime_a,diff2_cankle_Lx,'LineWidth',0.9,Color=[0 0 1 0.9]) %左
xlabel('Time [s]')
ylabel('左足首の加速度(補正前) [m/s^2]')
title('a')
set(gca, 'FontSize', 14); 
hold off

end

%ankle_y
display_ankle_y = false ; %true:表示 false:非表示
if display_ankle_y

figure
hold on
plot(ctime,cankle_Ry,'LineWidth',0.9,Color=[1 0 0 0.9]) %右
plot(ctime,cankle_Ly,'LineWidth',0.9,Color=[0 0 1 0.9]) %左
xlabel('Time [s]')
ylabel('DoTC')
title('ankle_y')
hold off

%加速度
figure
hold on
plot(ctime_a,diff2_cankle_Ry,'LineWidth',0.9,Color=[1 0 0 0.9]) %右
xlabel('Time [s]')
ylabel('a')
title('a')
hold off

%加速度
figure
hold on
plot(ctime_a,diff2_cankle_Ly,'LineWidth',0.9,Color=[0 0 1 0.9]) %左
xlabel('Time [s]')
ylabel('a')
title('a')
hold off

end

%hip_x
display_hip_x = false ; %true:表示 false:非表示
if display_hip_x

figure
hold on
plot(ctime,chip_Rx,'LineWidth',0.9,Color=[1 0 0 0.9]) %右
plot(ctime,chip_Lx,'LineWidth',0.9,Color=[0 0 1 0.9]) %左
xlabel('Time [s]')
ylabel('DoTC')
title('hip_x')
hold off

%加速度
figure
hold on
plot(ctime_a,diff2_chip_Rx,'LineWidth',0.9,Color=[1 0 0 0.9]) %右
xlabel('Time [s]')
ylabel('a')
title('a')
hold off

%加速度
figure
hold on
plot(ctime_a,diff2_chip_Lx,'LineWidth',0.9,Color=[0 0 1 0.9]) %左
xlabel('Time [s]')
ylabel('a')
title('a')
hold off

end

%hip_y
display_hip_y = false ; %true:表示 false:非表示
if display_hip_y

figure
hold on
plot(ctime,chip_Ry,'LineWidth',0.9,Color=[1 0 0 0.9]) %右
plot(ctime,chip_Ly,'LineWidth',0.9,Color=[0 0 1 0.9]) %左
xlabel('Time [s]')
ylabel('DoTC')
title('hip_y')
hold off

%加速度
figure
hold on
plot(ctime_a,diff2_chip_Ry,'LineWidth',0.9,Color=[1 0 0 0.9]) %右
xlabel('Time [s]')
ylabel('a')
title('a')
hold off

%加速度
figure
hold on
plot(ctime_a,diff2_chip_Ly,'LineWidth',0.9,Color=[0 0 1 0.9]) %左
xlabel('Time [s]')
ylabel('a')
title('a')
hold off

end

%knee_x
display_knee_x = false  ; %true:表示 false:非表示
if display_knee_x

figure
hold on
plot(ctime,cknee_Rx,'LineWidth',0.9,Color=[1 0 0 0.9]) %右
plot(ctime,cknee_Lx,'LineWidth',0.9,Color=[0 0 1 0.9]) %左
xlabel('Time [s]')
ylabel('DoTC')
title('knee_x')
hold off

%加速度
figure
hold on
plot(ctime_a,diff2_cknee_Rx,'LineWidth',0.9,Color=[1 0 0 0.9]) %右
xlabel('Time [s]')
ylabel('a')
title('a')
hold off

%加速度
figure
hold on
plot(ctime_a,diff2_cknee_Lx,'LineWidth',0.9,Color=[0 0 1 0.9]) %左
xlabel('Time [s]')
ylabel('a')
title('a')
hold off

end

%knee_y
display_knee_y = false  ; %true:表示 false:非表示
if display_knee_y

figure
hold on
plot(ctime,cknee_Ry,'LineWidth',0.9,Color=[1 0 0 0.9]) %右
plot(ctime,cknee_Ly,'LineWidth',0.9,Color=[0 0 1 0.9]) %左
xlabel('Time [s]')
ylabel('DoTC')
title('knee_y')
hold off

%加速度
figure
hold on
plot(ctime_a,diff2_cknee_Ry,'LineWidth',0.9,Color=[1 0 0 0.9]) %右
xlabel('Time [s]')
ylabel('a')
title('a')
hold off

%加速度
figure
hold on
plot(ctime_a,diff2_cknee_Ly,'LineWidth',0.9,Color=[0 0 1 0.9]) %左
xlabel('Time [s]')
ylabel('a')
title('a')
hold off

end

%bigtoe_x
display_bigtoe_x = false  ; %true:表示 false:非表示
if display_bigtoe_x

figure
hold on
plot(ctime,cbigtoe_Rx,'LineWidth',0.9,Color=[1 0 0 0.9]) %右
plot(ctime,cbigtoe_Lx,'LineWidth',0.9,Color=[0 0 1 0.9]) %左
xlabel('Time [s]')
ylabel('DoTC')
title('bigtoe_x')
hold off

%加速度
figure
hold on
plot(ctime_a,diff2_cbigtoe_Rx,'LineWidth',0.9,Color=[1 0 0 0.9]) %右
xlabel('Time [s]')
ylabel('右爪先の加速度(補正前) [m/s^2]')
xlim([4.5,6.7]);
ylim([-50 50]);
set(gca, 'FontSize', 14); 
hold off

%加速度
figure
hold on
plot(ctime_a,diff2_cbigtoe_Lx,'LineWidth',0.9,Color=[0 0 1 0.9]) %左
xlabel('Time [s]')
ylabel('左爪先の加速度(補正前) [m/s^2]')
xlim([4.5,6.7]);
ylim([-50 50]);
set(gca, 'FontSize', 14); 
hold off

end

%bigtoe_y
display_bigtoe_y = false  ; %true:表示 false:非表示
if display_bigtoe_y

figure
hold on
plot(ctime,cbigtoe_Ry,'LineWidth',0.9,Color=[1 0 0 0.9]) %右
plot(ctime,cbigtoe_Ly,'LineWidth',0.9,Color=[0 0 1 0.9]) %左
xlabel('Time [s]')
ylabel('DoTC')
title('bigtoe_y')
hold off

%加速度
figure
hold on
plot(ctime_a,diff2_cbigote_Ry,'LineWidth',0.9,Color=[1 0 0 0.9]) %右
xlabel('Time [s]')
ylabel('a')
title('a')
hold off

%加速度
figure
hold on
plot(ctime_a,diff2_cbigtoe_Ly,'LineWidth',0.9,Color=[0 0 1 0.9]) %左
xlabel('Time [s]')
ylabel('a')
title('a')
hold off

end

%heel_x
display_heel_x = false  ; %true:表示 false:非表示
if display_heel_x

figure
hold on
plot(ctime,cheel_Rx,'LineWidth',0.9,Color=[1 0 0 0.9]) %右
plot(ctime,cheel_Lx,'LineWidth',0.9,Color=[0 0 1 0.9]) %左
xlabel('Time [s]')
ylabel('DoTC')
title('heel_x')
hold off

%加速度
figure
hold on
plot(ctime_a,diff2_cheel_Rx,'LineWidth',0.9,Color=[1 0 0 0.9]) %右
xlabel('Time [s]')
ylabel('右踵の加速度(補正前) [m/s^2]')
xlim([4.5,6.7]);
ylim([-50 50]);
set(gca, 'FontSize', 14); 
hold off

%加速度
figure
hold on
plot(ctime_a,diff2_cheel_Lx,'LineWidth',0.9,Color=[0 0 1 0.9]) %左
xlabel('Time [s]')
ylabel('左踵の加速度(補正前) [m/s^2]')
xlim([4.5,6.7]);
ylim([-50 50]);
set(gca, 'FontSize', 14); 

end

%knee_y
display_heel_y = false  ; %true:表示 false:非表示
if display_heel_y

figure
hold on
plot(ctime,cheel_Ry,'LineWidth',0.9,Color=[1 0 0 0.9]) %右
plot(ctime,cheel_Ly,'LineWidth',0.9,Color=[0 0 1 0.9]) %左
xlabel('Time [s]')
ylabel('DoTC')
title('_y')
hold off

%加速度
figure
hold on
plot(ctime_a,diff2_cheel_Ry,'LineWidth',0.9,Color=[1 0 0 0.9]) %右
xlabel('Time [s]')
ylabel('a')
title('a')
hold off

%加速度
figure
hold on
plot(ctime_a,diff2_cheel_Ly,'LineWidth',0.9,Color=[0 0 1 0.9]) %左
xlabel('Time [s]')
ylabel('a')
title('a')
hold off

end

%% 加速度で検出エラーを判定後、カルマンフィルタで補正
[kankle_Lx,kankle_Rx] = kalman2(cankle_Lx,cankle_Rx,100,0.0005);     %(左座標,　右座標, 加速度の閾値, カルマンの初期値)　データによって閾値を変更
[kankle_Ly,kankle_Ry] = kalman2(cankle_Ly,cankle_Ry,500,0.003);
[kknee_Lx,kknee_Rx] = kalman2(cknee_Lx,cknee_Rx,100,0.008);
[kknee_Ly,kknee_Ry] = kalman2(cknee_Ly,cknee_Ry,500,0.002);
[khip_Lx,khip_Rx] = kalman2(chip_Lx,chip_Rx,100,0.005);
[khip_Ly,khip_Ry] = kalman2(chip_Ly,chip_Ry,500,0.003);
[kbigtoe_Lx,kbigtoe_Rx] = kalman2(cbigtoe_Lx,cbigtoe_Rx,100,0.008);
[kbigtoe_Ly,kbigtoe_Ry] = kalman2(cbigtoe_Ly,cbigtoe_Ry,500,0.003);
[kheel_Lx,kheel_Rx] = kalman2(cheel_Lx,cheel_Rx,100,0.08);
[kheel_Ly,kheel_Ry] = kalman2(cheel_Ly,cheel_Ry,500,0.003);


%% バターワースフィルタ

%4次元バターワースフィルタ sample100Hz cutoff 6Hz　
[kankle_Rx_filter,kankle_Lx_filter]=bufilter(kankle_Rx,kankle_Lx);
[kankle_Ry_filter,kankle_Ly_filter]=bufilter(kankle_Ry,kankle_Ly);
[kknee_Rx_filter,kknee_Lx_filter]=bufilter(kknee_Rx,kknee_Lx);
[kknee_Ry_filter,kknee_Ly_filter]=bufilter(kknee_Ry,kknee_Ly);
[khip_Rx_filter,khip_Lx_filter]=bufilter(khip_Rx,khip_Lx);
[khip_Ry_filter,khip_Ly_filter]=bufilter(khip_Ry,khip_Ly);
[kbigtoe_Rx_filter,kbigtoe_Lx_filter]=bufilter(kbigtoe_Rx,kbigtoe_Lx);
[kbigtoe_Ry_filter,kbigtoe_Ly_filter]=bufilter(kbigtoe_Ry,kbigtoe_Ly);
[kheel_Rx_filter,kheel_Lx_filter]=bufilter(kheel_Rx,kheel_Lx);
[kheel_Ry_filter,kheel_Ly_filter]=bufilter(kheel_Ry,kheel_Ly);


%% 関節角度計算

%補正前、補正後, バターワース後の足首・膝・股関節座標をもとに膝関節角度算出
before_knee_angle_R = kangle(cknee(:,1),chip(:,1),cankle(:,1),cknee(:,3),chip(:,3),cankle(:,3)); 
before_knee_angle_L = kangle(cknee(:,2),chip(:,2),cankle(:,2),cknee(:,4),chip(:,4),cankle(:,4));
after_knee_angle_R = kangle(kknee_Rx,khip_Rx,kankle_Rx,kknee_Ry,khip_Ry,kankle_Ry);
after_knee_angle_L = kangle(kknee_Lx,khip_Lx,kankle_Lx,kknee_Ly,khip_Ly,kankle_Ly);
bfilter_knee_angle_R = kangle(kknee_Rx_filter,khip_Rx_filter,kankle_Rx_filter, kknee_Ry_filter,khip_Ry_filter,kankle_Ry_filter);
bfilter_knee_angle_L = kangle(kknee_Lx_filter,khip_Lx_filter,kankle_Lx_filter, kknee_Ly_filter,khip_Ly_filter,kankle_Ly_filter);

%補正前、補正後の足首・膝・股関節座標をもとに足関節角度算出
before_ankle_angle_R = aangle(cknee(:,1),cankle(:,1),cbigtoe(:,1),cheel(:,1),cknee(:,3),cankle(:,3),cbigtoe(:,3),cheel(:,3));
before_ankle_angle_L = aangle(cknee(:,2),cankle(:,2),cbigtoe(:,2),cheel(:,2),cknee(:,4),cankle(:,4),cbigtoe(:,4),cheel(:,4));
after_ankle_angle_R = aangle(kknee_Rx,kankle_Rx,kbigtoe_Rx,kheel_Rx,kknee_Ry,kankle_Ry,kbigtoe_Ry,kheel_Ry);
after_ankle_angle_L = aangle(kknee_Lx,kankle_Lx,kbigtoe_Lx,kheel_Lx,kknee_Ly,kankle_Ly,kbigtoe_Ly,kheel_Ly);
bfilter_ankle_angle_R = aangle(kknee_Rx_filter,kankle_Rx_filter,kbigtoe_Rx_filter,kheel_Rx_filter,kknee_Ry_filter,kankle_Ry_filter,kbigtoe_Ry_filter,kheel_Ry_filter);
bfilter_ankle_angle_L = aangle(kknee_Lx_filter,kankle_Lx_filter,kbigtoe_Lx_filter,kheel_Lx_filter,kknee_Ly_filter,kankle_Ly_filter,kbigtoe_Ly_filter,kheel_Ly_filter);

%補正前、補正後の足首・膝・股関節座標をもとにこ股関節角度算出
before_hip_angle_R = hangle(chip(:,1),cknee(:,1),chip(:,3),cknee(:,3));
before_hip_angle_L = hangle(chip(:,2),cknee(:,2),chip(:,4),cknee(:,4));
after_hip_angle_R = hangle(khip_Rx,kknee_Rx,khip_Ry,kknee_Ry);
after_hip_angle_L = hangle(khip_Lx,kknee_Lx,khip_Ly,kknee_Ly);
bfilter_hip_angle_R = hangle(khip_Rx_filter,kknee_Rx_filter,khip_Ry_filter,kknee_Ry_filter);
bfilter_hip_angle_L = hangle(khip_Lx_filter,kknee_Lx_filter,khip_Ly_filter,kknee_Ly_filter);



%% 結果表示　座標

% 最初と最後の30個以外のデータを抽出(バターワースフィルタの影響をなくすため)
start_index = 31; % 31番目のデータから
end_index = length(ctime) - 30; % 最後から30個を除いたデータまで

%カルマンフィルタ補正後
display_before_butterworth = true ; %true:表示 false:非表示
if display_before_butterworth

%足首_x
figure
hold on
plot(ctime,cankle(:,1),'LineWidth',0.9,Color=[0 0 0 0.9]) %補正前右
plot(ctime,cankle(:,2),'LineWidth',0.9,Color=[0 0 0 0.9]) %補正前左
plot(ctime,kankle_Rx,'LineWidth',0.9,Color=[1 0 0 0.9]) %補正後右
plot(ctime,kankle_Lx,'LineWidth',0.9,Color=[0 0 1 0.9]) %補正後左
xlabel('Time [s]','FontSize', 16)
ylabel('DoTC','FontSize', 16)
title('ankle_x')
set(gca, 'FontSize', 14);  % フォントサイズ14に設定
hold off

end


%カルマンフィルタ＋バターワースフィルタ後
display_after_butterworth = false ;  %true:表示 false:非表示
if display_after_butterworth


%股関節_x
figure
hold on
plot(ctime(start_index:end_index),chip(start_index:end_index,1),'LineWidth',0.9,Color=[0 0 0 0.9]) %補正前右
plot(ctime(start_index:end_index),chip(start_index:end_index,2),'LineWidth',0.9,Color=[0 0 0 0.9]) %補正前左
plot(ctime(start_index:end_index),khip_Rx_filter(start_index:end_index),'LineWidth',0.9,Color=[1 0 0 0.9]) %補正後右
plot(ctime(start_index:end_index),khip_Lx_filter(start_index:end_index),'LineWidth',0.9,Color=[0 0 1 0.9]) %補正後左
xlabel('Time [s]','FontSize', 16)
ylabel('股関節のDoTC','FontSize', 16)
title('hip_x')
set(gca, 'FontSize', 14);  % フォントサイズ14に設定
hold off


%股関節_y
figure
hold on
plot(ctime(start_index:end_index),chip(start_index:end_index,3),'LineWidth',0.9,Color=[0 0 0 0.9]) %補正前右
plot(ctime(start_index:end_index),chip(start_index:end_index,4),'LineWidth',0.9,Color=[0 0 0 0.9]) %補正前左
plot(ctime(start_index:end_index),khip_Ry_filter(start_index:end_index),'LineWidth',0.9,Color=[1 0 0 0.9]) %補正後右
plot(ctime(start_index:end_index),khip_Ly_filter(start_index:end_index),'LineWidth',0.9,Color=[0 0 1 0.9]) %補正後左
xlabel('Time [s]','FontSize', 16)
ylabel('股関節のDoHC','FontSize', 16)
title('hip_y')
set(gca, 'FontSize', 14);  % フォントサイズ14に設定
hold off

%膝_x
figure
hold on
plot(ctime(start_index:end_index),cknee(start_index:end_index,1),'LineWidth',0.9,Color=[0 0 0 0.9]) %補正前右
plot(ctime(start_index:end_index),cknee(start_index:end_index,2),'LineWidth',0.9,Color=[0 0 0 0.9]) %補正前左
plot(ctime(start_index:end_index),kknee_Rx_filter(start_index:end_index),'LineWidth',0.9,Color=[1 0 0 0.9]) %補正後右
plot(ctime(start_index:end_index),kknee_Lx_filter(start_index:end_index),'LineWidth',0.9,Color=[0 0 1 0.9]) %補正後左
xlabel('Time [s]','FontSize', 16)
ylabel('膝のDoTC','FontSize', 16)
title(' knee_x')
set(gca, 'FontSize', 14);  % フォントサイズ14に設定
hold off

%膝_x
figure
hold on
plot(ctime(start_index:end_index),cknee(start_index:end_index,3),'LineWidth',0.9,Color=[0 0 0 0.9]) %補正前右
plot(ctime(start_index:end_index),cknee(start_index:end_index,4),'LineWidth',0.9,Color=[0 0 0 0.9]) %補正前左
plot(ctime(start_index:end_index),kknee_Ry_filter(start_index:end_index),'LineWidth',0.9,Color=[1 0 0 0.9]) %補正後右
plot(ctime(start_index:end_index),kknee_Ly_filter(start_index:end_index),'LineWidth',0.9,Color=[0 0 1 0.9]) %補正後左
xlabel('Time [s]','FontSize', 16)
ylabel('膝のDoHC','FontSize', 16)
title(' knee_y')
set(gca, 'FontSize', 14);  % フォントサイズ14に設定
hold off


%足首_x
figure
hold on
plot(ctime(start_index:end_index),cankle(start_index:end_index,1),'LineWidth',0.9,Color=[0 0 0 0.9]) %補正前右
plot(ctime(start_index:end_index),cankle(start_index:end_index,2),'LineWidth',0.9,Color=[0 0 0 0.9]) %補正前左
plot(ctime(start_index:end_index),kankle_Rx_filter(start_index:end_index),'LineWidth',0.9,Color=[1 0 0 0.9]) %補正後右
plot(ctime(start_index:end_index),kankle_Lx_filter(start_index:end_index),'LineWidth',0.9,Color=[0 0 1 0.9]) %補正後左
xlabel('Time [s]','FontSize', 16)
ylabel('足首のDoTC','FontSize', 16)
xlim([2,5.5]);
ylim([0,1400]);
%title('ankle_x')
set(gca, 'FontSize', 14);  % フォントサイズ14に設定
hold off

%足首_y
figure
hold on
plot(ctime(start_index:end_index),cankle(start_index:end_index,3),'LineWidth',0.9,Color=[0 0 0 0.9]) %補正前右
plot(ctime(start_index:end_index),cankle(start_index:end_index,4),'LineWidth',0.9,Color=[0 0 0 0.9]) %補正前左
plot(ctime(start_index:end_index),kankle_Ry_filter(start_index:end_index),'LineWidth',0.9,Color=[1 0 0 0.9]) %補正後右
plot(ctime(start_index:end_index),kankle_Ly_filter(start_index:end_index),'LineWidth',0.9,Color=[0 0 1 0.9]) %補正後左
xlabel('Time [s]','FontSize', 16)
ylabel('足首のDoHC','FontSize', 16)

title('ankle_y')
set(gca, 'FontSize', 14);  % フォントサイズ14に設定
hold off


%つま先
figure
hold on
plot(ctime(start_index:end_index),cbigtoe(start_index:end_index,1),'LineWidth',0.9,Color=[0 0 0 0.9]) %補正前右
plot(ctime(start_index:end_index),cbigtoe(start_index:end_index,2),'LineWidth',0.9,Color=[0 0 0 0.9]) %補正前左
plot(ctime(start_index:end_index),kbigtoe_Rx_filter(start_index:end_index),'LineWidth',0.9,Color=[1 0 0 0.9]) %補正後右
plot(ctime(start_index:end_index),kbigtoe_Lx_filter(start_index:end_index),'LineWidth',0.9,Color=[0 0 1 0.9]) %補正後左
xlabel('Time [s]','FontSize', 16)
ylabel('DoTC','FontSize', 16)
xlim([2,5.5]);
ylim([0,1400]);
title('bigtoe_x')
set(gca, 'FontSize', 14);  % フォントサイズ14に設定
hold off


%つま先_y
figure
hold on
plot(ctime(start_index:end_index),cbigtoe(start_index:end_index,3),'LineWidth',0.9,Color=[0 0 0 0.9]) %補正前右
plot(ctime(start_index:end_index),cbigtoe(start_index:end_index,4),'LineWidth',0.9,Color=[0 0 0 0.9]) %補正前左
plot(ctime(start_index:end_index),kbigtoe_Ry_filter(start_index:end_index),'LineWidth',0.9,Color=[1 0 0 0.9]) %補正後右
plot(ctime(start_index:end_index),kbigtoe_Ly_filter(start_index:end_index),'LineWidth',0.9,Color=[0 0 1 0.9]) %補正後左
xlabel('Time [s]','FontSize', 16)
ylabel('DoHC','FontSize', 16)
title('bigtoe_y')
set(gca, 'FontSize', 14);  % フォントサイズ14に設定
hold off

%かかと 
figure
hold on
plot(ctime(start_index:end_index),cheel(start_index:end_index,1),'LineWidth',0.9,Color=[0 0 0 0.9]) %補正前右
plot(ctime(start_index:end_index),cheel(start_index:end_index,2),'LineWidth',0.9,Color=[0 0 0 0.9]) %補正前左
plot(ctime(start_index:end_index),kheel_Rx_filter(start_index:end_index),'LineWidth',0.9,Color=[1 0 0 0.9]) %補正後右
plot(ctime(start_index:end_index),kheel_Lx_filter(start_index:end_index),'LineWidth',0.9,Color=[0 0 1 0.9]) %補正後左
xlabel('Time [s]','FontSize', 16)
ylabel('DoTC','FontSize', 16)
title('heel_x')
xlim([2,5.5]);
ylim([0,1400]);
set(gca, 'FontSize', 14);  % フォントサイズ14に設定
hold off

%かかと_y
figure
hold on
plot(ctime(start_index:end_index),cheel(start_index:end_index,3),'LineWidth',0.9,Color=[0 0 0 0.9]) %補正前右
plot(ctime(start_index:end_index),cheel(start_index:end_index,4),'LineWidth',0.9,Color=[0 0 0 0.9]) %補正前左
plot(ctime(start_index:end_index),kheel_Ry_filter(start_index:end_index),'LineWidth',0.9,Color=[1 0 0 0.9]) %補正後右
plot(ctime(start_index:end_index),kheel_Ly_filter(start_index:end_index),'LineWidth',0.9,Color=[0 0 1 0.9]) %補正後左
xlabel('Time [s]','FontSize', 16)
ylabel('DoHC','FontSize', 16)
title('heel_y')
set(gca, 'FontSize', 14);  % フォントサイズ14に設定
hold off

end

%% 結果表示　関節角度

%カルマンフィルタ＋バターワースフィルタ後
display_angle = false;
if display_angle %true:表示 false:非表示


%膝関節角度
figure
hold on
plot(ctime(start_index:end_index),bfilter_knee_angle_R(start_index:end_index),'LineWidth',1.0,'Color','#fc9d03')%提案法
plot(ctime(start_index:end_index),knee_cangle_R(start_index:end_index),'LineWidth',1.1,'Color','#42f58d')%3DMC
ylim([-30, 60]);
yline(0, '--', 'LineWidth', 1.0);
title('knee angle filter')
xlabel('Time [s]','FontSize', 16)
ylabel('Angle','FontSize', 16)
set(gca, 'FontSize', 14);  % フォントサイズ14に設定
hold off

%足関節角度
figure
hold on
plot(ctime(start_index:end_index),bfilter_ankle_angle_R(start_index:end_index),'LineWidth',1.0,'Color','#fc9d03')%提案法
plot(ctime(start_index:end_index),ankle_cangle_R(start_index:end_index),'LineWidth',1.1,'Color','#42f58d')%3DMC
title('feet angle filter')
ylim([-30, 60]);
yline(0, '--', 'LineWidth', 1.0);
xlabel('Time [s]','FontSize', 16)
ylabel('Angle','FontSize', 16)
set(gca, 'FontSize', 14);  % フォントサイズ14に設定
hold off

%股関節角度
figure
hold on
plot(ctime(start_index:end_index),bfilter_hip_angle_R(start_index:end_index),'LineWidth',1.0,'Color','#fc9d03')%提案法
plot(ctime(start_index:end_index),hip_cangle_R(start_index:end_index),'LineWidth',1.1,'Color','#42f58d')%3DMC
title('hip angle filter')
ylim([-30, 60]);
yline(0, '--', 'LineWidth', 1.0);
xlabel('Time [s]','FontSize', 16)
ylabel('Angle','FontSize', 16)
set(gca, 'FontSize', 14);  % フォントサイズ14に設定
hold off
 
end

%カルマンフィルタ＋バターワースフィルタ後　(±5度を表示)
display_angle_5 = true ; %true:表示 false:非表示
if display_angle_5

%膝関節角
%右
min_bfilter_knee_angle_R= bfilter_knee_angle_R-5; %関節角の-5を計算
wides = zeros(length(ctime(start_index:end_index)),1); 
wides = wides+10; 
min_knee_cangle_R=knee_cangle_R-5;

figure
hold on
plot(ctime(start_index:end_index),bfilter_knee_angle_R(start_index:end_index),'LineWidth',1.5,'Color','#fc9d03')%提案法オレンジ
plot(ctime(start_index:end_index),knee_cangle_R(start_index:end_index),'LineWidth',1.5,'Color','#42f58d')%3DMC緑

arA = area(ctime(start_index:end_index),[min_bfilter_knee_angle_R(start_index:end_index) wides]);%-5から+5までグラフを塗りつぶす
set(arA(1),'FaceColor','None','LineStyle',':','EdgeColor','#fc9d03')
set(arA(2),'FaceColor','#fc9d03','FaceAlpha',0.3,'LineStyle',':','EdgeColor','#fc9d03')

ar1 = area(ctime(start_index:end_index),[min_knee_cangle_R(start_index:end_index) wides]);%-5から+5までグラフを塗りつぶす
set(ar1(1),'FaceColor','None','LineStyle',':','EdgeColor','g')
set(ar1(2),'FaceColor','g','FaceAlpha',0.2,'LineStyle',':','EdgeColor','g')

xlabel('Time [s]','FontSize', 16)
ylabel('Knee Joint Angel_R [° ]','FontSize', 16)
set(gca, 'FontSize', 14);  % フォントサイズ14に設定
ylim([-10,80]);
hold off

%左
min_bfilter_knee_angle_L= bfilter_knee_angle_L-5;
wides = zeros(length(ctime(start_index:end_index)),1);
wides = wides+10;
min_knee_cangle_L=knee_cangle_L-5;

figure
hold on
plot(ctime(start_index:end_index),bfilter_knee_angle_L(start_index:end_index),'LineWidth',1.5,'Color','#fc9d03')%提案法
plot(ctime(start_index:end_index),knee_cangle_L(start_index:end_index),'LineWidth',1.5,'Color','#42f58d')%3DMC

arA = area(ctime(start_index:end_index),[min_bfilter_knee_angle_L(start_index:end_index) wides]);%-5から+5までグラフを塗りつぶす
set(arA(1),'FaceColor','None','LineStyle',':','EdgeColor','#fc9d03')
set(arA(2),'FaceColor','#fc9d03','FaceAlpha',0.3,'LineStyle',':','EdgeColor','#fc9d03')

ar1 = area(ctime(start_index:end_index),[min_knee_cangle_L(start_index:end_index) wides]);%-5から+5までグラフを塗りつぶす
set(ar1(1),'FaceColor','None','LineStyle',':','EdgeColor','g')
set(ar1(2),'FaceColor','g','FaceAlpha',0.2,'LineStyle',':','EdgeColor','g')

xlabel('Time [s]','FontSize', 16)
ylabel('Knee Joint Angel_L [° ]','FontSize', 16)
set(gca, 'FontSize', 14);  % フォントサイズ14に設定
ylim([-10,80]);
hold off




%足関節角度
%右
min_bfilter_ankle_angle_R= bfilter_ankle_angle_R-5;
min_ankle_cangle_R=ankle_cangle_R-5;

figure
hold on
plot(ctime(start_index:end_index),bfilter_ankle_angle_R(start_index:end_index),'LineWidth',1.5,'Color','#fc9d03')%提案法
plot(ctime(start_index:end_index),ankle_cangle_R(start_index:end_index),'LineWidth',1.5,'Color','#42f58d')%3DMC

arA = area(ctime(start_index:end_index),[min_bfilter_ankle_angle_R(start_index:end_index) wides]);%-5から+5までグラフを塗りつぶす
set(arA(1),'FaceColor','None','LineStyle',':','EdgeColor','#fc9d03')
set(arA(2),'FaceColor','#fc9d03','FaceAlpha',0.3,'LineStyle',':','EdgeColor','#fc9d03')

ar1 = area(ctime(start_index:end_index),[min_ankle_cangle_R(start_index:end_index) wides]);%-5から+5までグラフを塗りつぶす
set(ar1(1),'FaceColor','None','LineStyle',':','EdgeColor','g')
set(ar1(2),'FaceColor','g','FaceAlpha',0.2,'LineStyle',':','EdgeColor','g')

xlabel('Time [s]','FontSize', 16)
ylabel('Ankle Joint Angle_R [° ]','FontSize', 16)
set(gca, 'FontSize', 14);  % フォントサイズ14に設定
ylim([-30,30]);
hold off

%左
min_bfilter_ankle_angle_L= bfilter_ankle_angle_L-5;
min_ankle_cangle_L=ankle_cangle_L-5;

figure
hold on
plot(ctime(start_index:end_index),bfilter_ankle_angle_L(start_index:end_index),'LineWidth',1.5,'Color','#fc9d03')%提案法
plot(ctime(start_index:end_index),ankle_cangle_L(start_index:end_index),'LineWidth',1.5,'Color','#42f58d')%3DMC

arA = area(ctime(start_index:end_index),[min_bfilter_ankle_angle_L(start_index:end_index) wides]);%-5から+5までグラフを塗りつぶす
set(arA(1),'FaceColor','None','LineStyle',':','EdgeColor','#fc9d03')
set(arA(2),'FaceColor','#fc9d03','FaceAlpha',0.3,'LineStyle',':','EdgeColor','#fc9d03')

ar1 = area(ctime(start_index:end_index),[min_ankle_cangle_L(start_index:end_index) wides]);%-5から+5までグラフを塗りつぶす
set(ar1(1),'FaceColor','None','LineStyle',':','EdgeColor','g')
set(ar1(2),'FaceColor','g','FaceAlpha',0.2,'LineStyle',':','EdgeColor','g')

xlabel('Time [s]','FontSize', 16)
ylabel('Ankle Joint Angle_L [° ]','FontSize', 16)
set(gca, 'FontSize', 14);  % フォントサイズ14に設定
ylim([-30,30]);
hold off


%膝関節角度
%右
min_bfilter_hip_angle_R= bfilter_hip_angle_R-5;
min_hip_cangle_R=hip_cangle_R-5;

figure
hold on
plot(ctime(start_index:end_index),bfilter_hip_angle_R(start_index:end_index),'LineWidth',1.5,'Color','#fc9d03')%提案法
plot(ctime(start_index:end_index),hip_cangle_R(start_index:end_index),'LineWidth',1.5,'Color','#42f58d')%3DMC

arA = area(ctime(start_index:end_index),[min_bfilter_hip_angle_R(start_index:end_index) wides]);%-5から+5までグラフを塗りつぶす
set(arA(1),'FaceColor','None','LineStyle',':','EdgeColor','#fc9d03')
set(arA(2),'FaceColor','#fc9d03','FaceAlpha',0.3,'LineStyle',':','EdgeColor','#fc9d03')

ar1 = area(ctime(start_index:end_index),[min_hip_cangle_R(start_index:end_index) wides]);%-5から+5までグラフを塗りつぶす
set(ar1(1),'FaceColor','None','LineStyle',':','EdgeColor','g')
set(ar1(2),'FaceColor','g','FaceAlpha',0.2,'LineStyle',':','EdgeColor','g')

xlabel('Time [s]','FontSize', 16)
ylabel('Hip Joint Angle_R [° ]','FontSize', 16)
set(gca, 'FontSize', 14);  % フォントサイズ14に設定
ylim([-30,40]);
hold off

%左
min_bfilter_hip_angle_L= bfilter_hip_angle_L-5;
min_hip_cangle_L=hip_cangle_L-5;

figure
hold on
plot(ctime(start_index:end_index),bfilter_hip_angle_L(start_index:end_index),'LineWidth',1.5,'Color','#fc9d03')%提案法
plot(ctime(start_index:end_index),hip_cangle_L(start_index:end_index),'LineWidth',1.5,'Color','#42f58d')%3DMC

arA = area(ctime(start_index:end_index),[min_bfilter_hip_angle_L(start_index:end_index) wides]);%-5から+5までグラフを塗りつぶす
set(arA(1),'FaceColor','None','LineStyle',':','EdgeColor','#fc9d03')
set(arA(2),'FaceColor','#fc9d03','FaceAlpha',0.3,'LineStyle',':','EdgeColor','#fc9d03')

ar1 = area(ctime(start_index:end_index),[min_hip_cangle_L(start_index:end_index) wides]);%-5から+5までグラフを塗りつぶす
set(ar1(1),'FaceColor','None','LineStyle',':','EdgeColor','g')
set(ar1(2),'FaceColor','g','FaceAlpha',0.2,'LineStyle',':','EdgeColor','g')
ylim([-30,40]);
xlabel('Time [s]','FontSize', 16)
ylabel('Hip Joint Angle_L [° ]','FontSize', 16)
set(gca, 'FontSize', 14);  % フォントサイズ14に設定

hold off

end

%MAE表示
disp("右股関節角度のMAE:"+ mae(bfilter_hip_angle_R(start_index:end_index)-hip_cangle_R(start_index:end_index))); %股関節右
disp("右膝関節角度のMAE:"+mae(bfilter_knee_angle_R(start_index:end_index)-knee_cangle_R(start_index:end_index))); %膝右
disp("右足関節角度のMAE"+mae(bfilter_ankle_angle_R(start_index:end_index)-ankle_cangle_R(start_index:end_index)));%足右

disp("左股関節角度のMAE:"+mae(bfilter_hip_angle_L(start_index:end_index)-hip_cangle_L(start_index:end_index))); %股関節左
disp("左膝関節角度のMAE:"+mae(bfilter_knee_angle_L(start_index:end_index)-knee_cangle_L(start_index:end_index)));%膝左
disp("左足関節角度のMAE:"+mae(bfilter_ankle_angle_L(start_index:end_index)-ankle_cangle_L(start_index:end_index)));%足左





%% 二階差分カルマンフィルタ

function [cooredinate_L,coordinate_R] = kalman2(cooredinate_L,coordinate_R,th,initial_value)

%二階差分カルマンフィルタ
first_step = 1;
end_step = length(coordinate_R);
miss_point = zeros([end_step 1]);
%miss_point_cover=zeros([end_step 1]);
kalman_flag = 0;
for i = first_step+2:end_step
     diff_data_Lx = diff(cooredinate_L(1:i,:));
     MAF_diff_Lx = maf(diff_data_Lx,3);
     yL = MAF_diff_Lx;
     diff_data_Rx = diff(coordinate_R(1:i,:));
     MAF_diff_Rx = maf(diff_data_Rx,3);
     yR = MAF_diff_Rx;
     Len = length(yL);


           %最尤推定よりパラメータを求める
        parL = initial_value;
        parR = initial_value;
        varEtaL = parL; % σ^2_η の初期値
        varEpsL = parL; % σ^2_ε の初期値
        varEtaR = parR; % σ^2_η の初期値
        varEpsR = parR; % σ^2_ε の初期値
    
        psiEtaL = log(sqrt(varEtaL)); % ψ_η に変換
        psiEpsL = log(sqrt(varEpsL)); % ψ_ε に変換
        psiEtaR = log(sqrt(varEtaR)); % ψ_η に変換
        psiEpsR = log(sqrt(varEpsR)); % ψ_ε に変換
    

        options = optimoptions(@fminunc,'Display','off','Algorithm','quasi-newton','CheckGradients',true,'FiniteDifferenceType','central'); % 準ニュートン法
        x0L = [psiEtaL, psiEpsL];              % 探索するパラメータの初期値
        x0R = [psiEtaR, psiEpsR];              % 探索するパラメータの初期値
        fL = @(xL) -calcLogDiffuseLlhd(yL, xL); % 最小化したい函数（散漫な対数尤度の最大化なので負号をつける）
        fR = @(xR) -calcLogDiffuseLlhd(yR, xR); % 最小化したい函数（散漫な対数尤度の最大化なので負号をつける）        
        xoptL = fminunc(fL, x0L, options);     % 実行
        xoptR = fminunc(fR, x0R, options);     % 実行
    
        varEtaOptL = exp(2*xoptL(1));          % 推定されたψ_ηをσ^2_ηに戻す
        varEpsOptL = exp(2*xoptL(2));          % 推定されたψ_εをσ^2_εに戻す
        varEtaOptR = exp(2*xoptR(1));          % 推定されたψ_ηをσ^2_ηに戻す
        varEpsOptR = exp(2*xoptR(2));          % 推定されたψ_εをσ^2_εに戻す
    
        %変更後パラメータの代入
        varEpsL = varEtaOptL;
        varEtaL = varEpsOptL;
        varEpsR = varEtaOptR;
        varEtaR = varEpsOptR;
         
        a1L = varEpsL;
        P1L = varEtaL;
        a1R = varEpsR;
        P1R = varEtaR;
        % カルマンフィルタの初期値
        a_tt1L = zeros([length(yL)+1 1]); a_tt1L(1) = a1L;
        P_tt1L = zeros([length(yL)+1 1]); P_tt1L(1) = P1L;
        v_tL   = zeros([length(yL) 1]);
        F_tL   = zeros([length(yL) 1]); 
        a_ttL  = zeros([length(yL) 1]); 
        P_ttL  = zeros([length(yL) 1]);
        K_tL   = zeros([length(yL) 1]);
    
        a_tt1R = zeros([length(yR)+1 1]); a_tt1R(1) = a1R;
        P_tt1R = zeros([length(yR)+1 1]); P_tt1R(1) = P1R;
        v_tR   = zeros([length(yR) 1]); 
        F_tR   = zeros([length(yR) 1]); 
        a_ttR  = zeros([length(yR) 1]); 
        P_ttR  = zeros([length(yR) 1]);
        K_tR   = zeros([length(yR) 1]);
        
        for t = 1:Len
    
            % Innovation
            v_tL(t) = yL(t) - a_tt1L(t);
            v_tR(t) = yR(t) - a_tt1R(t);
    
            F_tL(t) = P_tt1L(t) + varEpsL;
            F_tR(t) = P_tt1R(t) + varEpsR;            
            
            % Kalman gain
            K_tL(t) = P_tt1L(t)/F_tL(t);
            K_tR(t) = P_tt1R(t)/F_tR(t);
    
            % Current state %L:1 R:2 L-R:3
            a_ttL(t) = a_tt1L(t) + K_tL(t)*v_tL(t);
            a_ttR(t) = a_tt1R(t) + K_tR(t)*v_tR(t);
            P_ttL(t) = P_tt1L(t) * (1 - K_tL(t));
            P_ttR(t) = P_tt1R(t) * (1 - K_tR(t));
    
            % Next state
            a_tt1L(t+1) = a_ttL(t);
            P_tt1L(t+1) = P_ttL(t) + varEtaL;
            a_tt1R(t+1) = a_ttR(t);
            P_tt1R(t+1) = P_ttR(t) + varEtaR;
    
        end

%加速度で検出エラーの種類を判別後、補正
    q1L = cooredinate_L(i)-cooredinate_L(i-1);  q2L = cooredinate_L(i-1)-cooredinate_L(i-2);
    q1R = coordinate_R(i)-coordinate_R(i-1);  q2R = coordinate_R(i-1)-coordinate_R(i-2);
    diff2_cankle_Lx_update= q1L-q2L;
    diff2_cankle_Rx_update= q1R-q2R;


 

    if abs(diff2_cankle_Lx_update)>th && abs(diff2_cankle_Rx_update)>th   %入れ替わりor両方誤検出 
        Lbox = cooredinate_L(i);  Rbox = coordinate_R(i); 
        cooredinate_L(i)=Rbox; coordinate_R(i)=Lbox;
        
        q1L = cooredinate_L(i)-cooredinate_L(i-1);  q2L = cooredinate_L(i-1)-cooredinate_L(i-2);
        q1R = coordinate_R(i)-coordinate_R(i-1);  q2R = coordinate_R(i-1)-coordinate_R(i-2);
        diff2_cankle_Lx_update= q1L-q2L;
        diff2_cankle_Rx_update= q1R-q2R;

        if abs(diff2_cankle_Lx_update)>th && abs(diff2_cankle_Rx_update)>th  
            Lbox = cooredinate_L(i);  Rbox = coordinate_R(i); 
            cooredinate_L(i)=Rbox; coordinate_R(i)=Lbox;
            cooredinate_L(i) = cooredinate_L(i-1) + a_tt1L(end-1);
            coordinate_R(i) = coordinate_R(i-1) + a_tt1R(end-1);
            miss_point(i)=4; %両方誤検出=4
            kalman_flag=1;

        else

            miss_point(i)=1; %入れ替わり=1  
            kalman_flag=1;

        end
    
    end
    if abs(diff2_cankle_Lx_update)> th && abs(diff2_cankle_Rx_update)<= th % 左ミス(重複と誤検出)
       cooredinate_L(i) = cooredinate_L(i-1) + a_tt1L(end-1);
       miss_point(i)=2; %左ミス=2
         kalman_flag=1;
    end    

    if abs(diff2_cankle_Lx_update)<= th && abs(diff2_cankle_Rx_update)> th  % 右ミス(重複と誤検出)
       coordinate_R(i) = coordinate_R(i-1) + a_tt1R(end-1);
       miss_point(i)=3; %右ミス=3
       kalman_flag=1;

    end


    p1L = cooredinate_L(i)-cooredinate_L(i-1);  p2L = cooredinate_L(i-1)-cooredinate_L(i-2);
    p1R = coordinate_R(i)-coordinate_R(i-1);  p2R = coordinate_R(i-1)-coordinate_R(i-2);
    diff2_cankle_Lx_update_cover= p1L-p2L;
    diff2_cankle_Rx_update_cover= p1R-p2R;


    th_cover=500;%
    if abs(diff2_cankle_Lx_update_cover)>=th_cover &&  kalman_flag==1
        cooredinate_L(i) = cooredinate_L(i-1)+p2L;
    end
    if abs(diff2_cankle_Rx_update_cover)>=th_cover &&  kalman_flag==1
        coordinate_R(i) = coordinate_R(i-1)+p2R;
      
    end
    end
        kalman_flag=0;

end








%% 使用関数一覧

%バターワースフィルタ
function [fcoordinate_Rx,fcoordinate_Lx] = bufilter(coordinate_R,coordinate_L)

order = 4; %(4次)
fs = 100;
fc = 6;                    % カットオフ周波数 (Hz)
[b, a] = butter(order, fc/(fs/2));   % 正規化カットオフ
fcoordinate_Rx = filtfilt(b, a, coordinate_R);       % ゼロ位相フィルタ（前後に通す）
fcoordinate_Lx = filtfilt(b, a, coordinate_L); 

end

% 膝関節角度算出
function angle = kangle(Xk,Xhi,Xa,Yk,Yhi,Ya)
    vA = (Xk-Xhi).*(Xk-Xa);
    vB = (Yk-Yhi).*(Yk-Ya);
    
    vAvB = vA+vB;
    
    Asize = sqrt((Xk-Xhi).^2+(Yk-Yhi).^2);
    Bsize = sqrt((Xk-Xa).^2+(Yk-Ya).^2);
    
    angle = 180-(180./pi.*acos(vAvB./(Asize.*Bsize)));

end


% 足関節角度算出

function angle = aangle(Xk,Xa,Xb,Xhe,Yk,Ya,Yb,Yhe)

     vC = (Xk-Xa).*(Xb-Xhe);
     vD = (Yk-Ya).*(Yb-Yhe);

     vCvD= vC+vD;

     Csize = sqrt((Xk-Xa).^2+(Yk-Ya).^2);
     Dsize = sqrt((Xb-Xhe).^2+(Yb-Yhe).^2);

     angle = 90-(180./pi.*acos(vCvD./(Csize.*Dsize)));
end

% 股関節角度
function angle = hangle(Xhi,Xk,Yhi,Yk)
Esize = sqrt((Xk-Xhi).^2+(Yk-Yhi).^2);
Ey=Yk-Yhi;
  
 hip_angle = 180-(180./pi.*acos(-Ey./Esize));
  for i = 1:length(Xhi)
         if Xhi(i) >= Xk(i)
            angle_m(i) = hip_angle(i);
         else
            angle_m(i) = -1* hip_angle(i);
         end
  end
    angle = -angle_m';
end


% 移動平均フィルタ
function y = maf(input,size)
    windowSize = size; 
    b = (1/windowSize)*ones(1,windowSize);
    a = 1;
    y = filter(b,a,input);
end

function logLd = calcLogDiffuseLlhd(y, vars)
% ローカルトレンドモデルの散漫な対数尤度を求める函数.
% y   : データ
% vars: 尤度に関わるパラメータ（ψ_ε, ψ_η）
 
psiEta = vars(1); varEta = exp(2*psiEta); % σ^2_η に戻す
psiEps = vars(2); varEps = exp(2*psiEps); % σ^2_ε に戻す
L = length(y);
 
% a_1, P_1の初期値 式（dissufe init.）参照のこと
a1 = y(1); 
P1 = varEps;
 
% カルマンフィルタリング
[~, ~, F_t, v_t] = localTrendKF(y, a1, P1, varEta, varEps); 
 
% 散漫対数尤度を計算
tmp = sum(log(F_t(2:end)) + v_t(2:end).^2 ./ F_t(2:end));
logLd = -0.5*L*log(2*pi) - 0.5 * tmp;
 
end

function [a_tt, P_tt, F_t, v_t] = localTrendKF(y, a1, P1, varEta, varEps)


% ローカルトレンドモデルのカルマンフィルタリングを行う函数
 L = length(y);
 
% Preallocation
a_tt1 = zeros(size(y)); a_tt1(1) = a1;
P_tt1 = zeros(size(y)); P_tt1(1) = P1;
v_t   = zeros(size(y)); 
F_t   = zeros(size(y)); 
a_tt  = zeros(size(y)); 
P_tt  = zeros(size(y));
K_t   = zeros(size(y));
 
% Filtering
for t = 1:L
    
    % Innovation
    v_t(t) = y(t) - a_tt1(t);
    F_t(t) = P_tt1(t) + varEps;
    
    % Kalman gain
    K_t(t) = P_tt1(t)/F_t(t);
 
    % Current state
    a_tt(t) = a_tt1(t) + K_t(t)*v_t(t);
    P_tt(t) = P_tt1(t) * (1 - K_t(t));
 
    % Next state
    a_tt1(t+1) = a_tt(t);
    P_tt1(t+1) = P_tt(t) + varEta;
    
end
end