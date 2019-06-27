//
// Created by alex on 16-5-25.
//

#ifndef DENSEFLOW_WARP_FLOW_H_H
#define DENSEFLOW_WARP_FLOW_H_H

cv::Mat windowedMatchingMask( const std::vector<cv::KeyPoint>& keypoints1, const std::vector<cv::KeyPoint>& keypoints2,
                          float maxDeltaX, float maxDeltaY )
{
  if( keypoints1.empty() || keypoints2.empty() )
    return cv::Mat();

  int n1 = (int)keypoints1.size(), n2 = (int)keypoints2.size();
  cv::Mat mask( n1, n2, CV_8UC1 );
  for( int i = 0; i < n1; i++ )
    {
      for( int j = 0; j < n2; j++ )
        {
          cv::Point2f diff = keypoints2[j].pt - keypoints1[i].pt;
          mask.at<uchar>(i, j) = std::abs(diff.x) < maxDeltaX && std::abs(diff.y) < maxDeltaY;
        }
    }
  return mask;
}

void MyWarpPerspective(Mat& prev_src, Mat& src, Mat& dst, Mat& M0, int flags=INTER_LINEAR,
                       int borderType=BORDER_CONSTANT, const Scalar& borderValue=Scalar())
{
    int width = src.cols;
    int height = src.rows;
    dst.create( height, width, CV_8UC1 );

    Mat mask = Mat::zeros(height, width, CV_8UC1);
    const int margin = 5;

    const int BLOCK_SZ = 32;
    short XY[BLOCK_SZ*BLOCK_SZ*2], A[BLOCK_SZ*BLOCK_SZ];

    int interpolation = flags & INTER_MAX;
    if( interpolation == INTER_AREA )
        interpolation = INTER_LINEAR;

    double M[9];
    Mat matM(3, 3, CV_64F, M);
    M0.convertTo(matM, matM.type());
    if( !(flags & WARP_INVERSE_MAP) )
        invert(matM, matM);

    int x, y, x1, y1;

    int bh0 = min(BLOCK_SZ/2, height);
    int bw0 = min(BLOCK_SZ*BLOCK_SZ/bh0, width);
    bh0 = min(BLOCK_SZ*BLOCK_SZ/bw0, height);

    for( y = 0; y < height; y += bh0 ) {
        for( x = 0; x < width; x += bw0 ) {
            int bw = min( bw0, width - x);
            int bh = min( bh0, height - y);

            Mat _XY(bh, bw, CV_16SC2, XY);
            Mat matA;
            Mat dpart(dst, Rect(x, y, bw, bh));

            for( y1 = 0; y1 < bh; y1++ ) {

                short* xy = XY + y1*bw*2;
                double X0 = M[0]*x + M[1]*(y + y1) + M[2];
                double Y0 = M[3]*x + M[4]*(y + y1) + M[5];
                double W0 = M[6]*x + M[7]*(y + y1) + M[8];
                short* alpha = A + y1*bw;

                for( x1 = 0; x1 < bw; x1++ ) {

                    double W = W0 + M[6]*x1;
                    W = W ? INTER_TAB_SIZE/W : 0;
                    double fX = max((double)INT_MIN, min((double)INT_MAX, (X0 + M[0]*x1)*W));
                    double fY = max((double)INT_MIN, min((double)INT_MAX, (Y0 + M[3]*x1)*W));

                    double _X = fX/double(INTER_TAB_SIZE);
                    double _Y = fY/double(INTER_TAB_SIZE);

                    if( _X > margin && _X < width-1-margin && _Y > margin && _Y < height-1-margin )
                        mask.at<uchar>(y+y1, x+x1) = 1;

                    int X = saturate_cast<int>(fX);
                    int Y = saturate_cast<int>(fY);

                    xy[x1*2] = saturate_cast<short>(X >> INTER_BITS);
                    xy[x1*2+1] = saturate_cast<short>(Y >> INTER_BITS);
                    alpha[x1] = (short)((Y & (INTER_TAB_SIZE-1))*INTER_TAB_SIZE + (X & (INTER_TAB_SIZE-1)));
                }
            }

            Mat _matA(bh, bw, CV_16U, A);
            remap( src, dpart, _XY, _matA, interpolation, borderType, borderValue );
        }
    }

    for( y = 0; y < height; y++ ) {
        const uchar* m = mask.ptr<uchar>(y);
        const uchar* s = prev_src.ptr<uchar>(y);
        uchar* d = dst.ptr<uchar>(y);
        for( x = 0; x < width; x++ ) {
            if(m[x] == 0)
                d[x] = s[x];
        }
    }
}

void ComputeMatch(const std::vector<KeyPoint>& prev_kpts, const std::vector<KeyPoint>& kpts,
                  const Mat& prev_desc, const Mat& desc, std::vector<Point2f>& prev_pts, std::vector<Point2f>& pts)
{
    prev_pts.clear();
    pts.clear();

    if(prev_kpts.size() == 0 || kpts.size() == 0)
        return;

    Mat mask = windowedMatchingMask(kpts, prev_kpts, 25, 25);

    BFMatcher desc_matcher(NORM_L2);
    std::vector<DMatch> matches;

    desc_matcher.match(desc, prev_desc, matches, mask);

    prev_pts.reserve(matches.size());
    pts.reserve(matches.size());

    for(size_t i = 0; i < matches.size(); i++) {
        const DMatch& dmatch = matches[i];
        // get the point pairs that are successfully matched
        prev_pts.push_back(prev_kpts[dmatch.trainIdx].pt);
        pts.push_back(kpts[dmatch.queryIdx].pt);
    }

    return;
}

void MergeMatch(const std::vector<Point2f>& prev_pts1, const std::vector<Point2f>& pts1,
                const std::vector<Point2f>& prev_pts2, const std::vector<Point2f>& pts2,
                std::vector<Point2f>& prev_pts_all, std::vector<Point2f>& pts_all)
{
    prev_pts_all.clear();
    prev_pts_all.reserve(prev_pts1.size() + prev_pts2.size());

    pts_all.clear();
    pts_all.reserve(pts1.size() + pts2.size());

    for(size_t i = 0; i < prev_pts1.size(); i++) {
        prev_pts_all.push_back(prev_pts1[i]);
        pts_all.push_back(pts1[i]);
    }

    for(size_t i = 0; i < prev_pts2.size(); i++) {
        prev_pts_all.push_back(prev_pts2[i]);
        pts_all.push_back(pts2[i]);
    }

    return;
}

void MatchFromFlow(const Mat& prev_grey, const Mat& flow, std::vector<Point2f>& prev_pts, std::vector<Point2f>& pts, const Mat& mask)
{
    int width = prev_grey.cols;
    int height = prev_grey.rows;
    prev_pts.clear();
    pts.clear();

    const int MAX_COUNT = 1000;
    goodFeaturesToTrack(prev_grey, prev_pts, MAX_COUNT, 0.001, 3, mask);

    if(prev_pts.size() == 0)
        return;

    for(int i = 0; i < prev_pts.size(); i++) {
        int x = std::min<int>(std::max<int>(cvRound(prev_pts[i].x), 0), width-1);
        int y = std::min<int>(std::max<int>(cvRound(prev_pts[i].y), 0), height-1);

        const float* f = flow.ptr<float>(y);
        pts.push_back(Point2f(x+f[2*x], y+f[2*x+1]));
    }
}

void MatchFromFlow_copy(const Mat& prev_grey, const Mat& flow_x, const Mat& flow_y, std::vector<Point2f>& prev_pts, std::vector<Point2f>& pts, const Mat& mask)
{
    int width = prev_grey.cols;
    int height = prev_grey.rows;
    prev_pts.clear();
    pts.clear();

    const int MAX_COUNT = 1000;
    goodFeaturesToTrack(prev_grey, prev_pts, MAX_COUNT, 0.001, 3, mask);

    if(prev_pts.size() == 0)
        return;

    for(int i = 0; i < prev_pts.size(); i++) {
        int x = std::min<int>(std::max<int>(cvRound(prev_pts[i].x), 0), width-1);
        int y = std::min<int>(std::max<int>(cvRound(prev_pts[i].y), 0), height-1);

        const float* f_x = flow_x.ptr<float>(y);
        const float* f_y = flow_y.ptr<float>(y);
        pts.push_back(Point2f(x+f_x[x], y+f_y[y]));
    }
}

#endif //DENSEFLOW_WARP_FLOW_H_H
