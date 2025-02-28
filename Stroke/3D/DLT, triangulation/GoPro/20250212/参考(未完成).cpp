char buf [1024];
1r = 1 % 2;
if( lr == 0) found[0] = found [1] = false;
if !fgets( buf, sizeof(buf)-3, f )) break;
size_t len = stren (buf) ;
while( len > 0 && isspace(buf [len-1])) buf [--len] = '10';
if buf[0] == '#') continue;
CV::Mat img = cv:: imread ( buf, 0 );
if img. empty() ) break;
imageSize = img. size();
imageNames [r]. push_back(buf) ;
i++;
11左の画像でボードを見つけられなかった場合、
11右の画像でそれを見つける意味はない
1/
if lr = 1 && !found [0] )
continue;
// サークルグリッドとその中心を見つける：
for ( int s = s s < maxScale; s++) {
cv:: Mat timg = img;
if(s >1)
resize img, timg, cv:: Size(), s,
S,
cv:: INTER_ CUBIC);
found [r] = cv:: findCirclesGrid(
timg,
CV:: Size(nx, ny),
corners [r],
cv:: CALIB_CB_ASYMMETRIC_GRID | cv:: CALIB_CB_CLUSTERING
）：
if found[r] || s == maxScale ) {
cv:: Mat mcorners ( corners[lr] );
corners *= (1./s) ;
if ( found[r] ) break;
}
if ( displayCorners ) f
cout < buf < endl; cv:: Mat cimg;
cv:: cvtColor (img, cimg, cv:: COLOR_GRAYBGR );
//チェスボードのコーナーの描画関数がサークルグリッドにも使える
cv:: drawChessboardCorners (
cimg, cv:: Size(nx, ny), corners[r], found [lr]
）;
CV:: imshow( "Corners", cimg );



19.4 ステレオ画像処理/125
まし IcV:iMaitkey 10） 8255）== 27.） 11 ESCキーで終了する
exit （-1）;
else
cout <^;
if( 1r = 1 && found[0] && found (11 ) {
objectPoints. push_back(boardModel);
points [0] .push_back(corners [0]): points [11 .push_back(corners [1]);
fclose (f);
//ステレオカメラのキャリブレーション
cv::Mat M1 = cV::Mat::eye（3,3,CV_64F）;
cv::Mat M2 = CV::Mat::eye（ 3,3, CV_64F）;
CV::Mat D1, D2, R, T,E,F;
cout <<"\nRunning stereo calibration ...\n";
cv:: stereoCalibrate(
objectPoints, points [0], points [1],
M1, D1, M2, D2,
imageSize, R, T, E, F, CV:: TermCriteria(
Cv: :TermCriteria:: COUNT | cv:: TermCriteria: :EPS, 100, le-5
）.
CV: : CALIB_FIX ASPECT_RATIO
I Cv: : CALIB_ZERO_TANGENT_DIST | cv:: CALIB_SAME_FOCAL_LENGTH

cout <<"Done\n\n";
11キャリブレーション精度チェック
11出力用の基礎行列にはすべての出力情報が含まれているため、!/エピポーラ幾何物束（m2^t*F*m1=0）を使用してキャリブレーションの品質を確認できる
vector< cv:: Point3f > lines [2];
double avgErr = 0;
int nframes = (int) objectPoints.size();
for ( i = 0; i < nframes; i++ ) {
vector< cv:: Point2f >& pt0 = points[0][1);
vector< cv: :Point2f >& pt1 = points [1) [1];
CV: :undistortPoints( pt0, pto, M1, D1, Cv::Mat(), ML );
Cv: :undistortPoints( pt1, pt1, M2, D2,
Cv: :Mat (), M2 );
cv: : computeCorrespondEpilines (ptO, 1, F, Lines(01 );



72619章
射影変換と 3次元ビジョン
cv:: computeCorrespondEpilines( ptl, ptl, F, lines lines(11);
for ( j = 0; j < N; j++ ) {
double err = fabs(
pt0ljl.x*lines(1l[jl.x + ptoljl.y+lines[1lljl.y + lines[lllil.z
) + fabs (
pt1[j1.x*Lines101(j1.x + pt1[jl.y*Lines(0](jl.y + Lines(0]0J1.z
）;
avgErr += err;
cout <
"avg
err = " « avgErr/(nframes*N) << endl;
11平行化の計算、および表示
//
if( showUndistorted ) {
cv: :Mat R1, R2, P1, P2, map11, map12, map21, map22;
11 キャリブレーションされている場合（Bouguet 法）
if( !useUncalibrated ) {
stereoRectify(
M1, D1, M2, D2, imageSize,
R, T, R1, R2, P1, P2,
Cv: : noArray(), 0
fabs (P2. at<double> (1, 3)) > fabs (P2.at<double> (0, 3)):
//cv::remap（）用にマップを事前計算する
initUndistortRectifyMap (
M1, D1, R1, Pl, imageSize, CV_16SC2, map11, map12
initUndistortRectifyMap (
M2, D2, R2, P2, imageSize, CV_16SC2, map21, map22
）;
11 キャリブレーションされていない場合（Hartley 法）
else ｛
1/各カメラの内部パラメータを使用するが、
11基礎行列から直接変換を計算する
vectors cv::Point2f > allpoints ［2］;
for ( i = 0; i < nframes; i++ ) {
copy (
points [0] [i].begin(),
points [0][i]. end(), back inserter(allpoints[0])
）;
copy (



    19.4ステレオ画像処理 727
    points (11[1). begin(). points [11(1].end(),
    back inserter (allpoints(11)
    CV:: Mat F = findFundamentalMat (
    allpoints(0], allpoints(1], cv:: :FM_8POINT
    ）：
    cv:: Mat H1, H2;
    cv: stereoRectifyUncalibrated(
    allpoints [0], allpoints(1],
    F,
    imageSize, H1, H2,
    R1 = M1. inv() *H1*M1;
    R2 = M2. inv ()*H2*M2;
    1/CV：;remap（）のための事前計算マップ
    1/
    cv: :initUndistortRectifyMap(
    M1, D1, R1, P1, imageSize, CV_16SC2,
    mapll, map12
    ）;
    cv:: initUndistortRectifyMap(
    M2, D2, R2, P2, imageSize, CV_16SC2,
    map21, map22
    ）;
    11画像を平行化し、視差マップを求める
    1/
    cv:: Mat pair;
    if lisVerticalStereo)
    pair.create( imageSize.height, imageSize.width*2, CV_8UC3 ); pair. create( imageSize.height*2, imageSize.width, CV_8UC3 );
    11ステレオ対応を見つけるためのセットアップ
    CV::Ptr<cv:: StereoSGBM> stereo = cv: :StereoSGBM:: create(
    -64,128,
    11,100, 1000,
    32,
    O,
    15, 1000, 16,
    StereoSGBM: : MODE_HH
    ）;
    for i = 0;
    i < nframes; i++ ) {



        Cv:: Mat imgl = Cv:: imread ( imageNames imageNames c_str(), 0 ):
        Cv:: Mat img2 = cv:: imread ( imageNames imageNames [1]. C_str(), 0 );
        cv:: Mat imglr,
        img2r, disp, disp;
        if imgl.empty() || img2.empty() )
        continue;
        cv:: remap( imgl,
        imglr, mapll, map12, cv:: INTER_LINEAR ):
        cv: : INTER_LINEAR );
        if（ !isVerticalstereo II !useUncalibrated ）｛
        11ステレオカメラが垂直に配置されている場合、1/ Hartley 法は画像を転置しないので、
        11平行化された画像のエピポーラ線は垂直である。
        // ステレオ対応づけ機能はこのような場合には対応していない
        stereo-›compute( imgir, img2r, disp); cv:: normalize disp,
        256, Cv:: NORM_MINMAX, CV_8U ):
        cv:: imshow( "disparity", vdisp ):
        if !isVerticalStereo)
        Cv:: Mat part = pair.colRange(0, imageSize.width);
        cvtColor (imgir, part, cv:: COLOR_GRAY2BGR) ;
        part = pair.colRange( imageSize.width, imageSize.width*2 );
        cvtColor (img2r, part, cv:: COLOR_GRAY2BGR) ;
        for (j = 0; j< imageSize.height; j += 16 )
        cv:: line(
        cv:: Point (0,j),
        Cv:: Point (imageSize.width*2,j),
        cv:: Scalar (0,255,0)
        ）;
        Cv::Mat part = pair. pair.rowRange(0, imageSize.height);
        cv:: cvtColor (imglr, part, cv:: cv::COLOR_GRAY2BGR. );
        part = pair. rowRange imageSize.height, imageSize.height*2 );
        cv: : cvtColor( img2r, part, cv:: COLOR_GRAY2BGR ) ;
        for (j = 0; j< imageSize.width; j += 16 )
        line( pair,
        cv: : Point (j,0),
        cv: :Point(j, imageSize.height*2),
        cv:: Scalar (0,255,0)
        }
        ）;
        cv:: imshow( "rectified", pair ); if (cv::waitKey ()&255) == 27 )
        break;

        int main (int argc, char** argv) {
            help ( argv );
            int board_w = 9, board_h = 6;
            const char* board_list = "ch12_list.txt";
            if( argc = 4 ) {
            board_ list = argv[1];
            board_w = atoi( argv[2] );
            board_h = atoi( argv[3] );
            }
            StereoCalib (board_list, board_w, board_h, true); return 0;