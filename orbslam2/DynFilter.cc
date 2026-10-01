// [study] Dynamic-point filters (FLOW, GEOM) and per-frame evaluation logging for ORB-SLAM2.
// Written for "Does Dynamic-Point Filtering Help When Texture Is Scarce?".
// The ground-truth mask is used ONLY for logging decisions; it never influences tracking.
#include "Tracking.h"
#include "Frame.h"
#include "KeyFrame.h"
#include "MapPoint.h"
#include "Map.h"
#include "System.h"
#include <opencv2/video/tracking.hpp>
#include <opencv2/calib3d.hpp>
#include <deque>
#include <cmath>
#include <algorithm>

namespace ORB_SLAM2
{

// ---------------------------------------------------------------------------------------------
void Frame::RemoveKeypoints(const std::vector<bool> &vbRemove)
{
    std::vector<int> keep; keep.reserve(N);
    for(int i=0;i<N;i++) if(!vbRemove[i]) keep.push_back(i);
    const int M = (int)keep.size();
    std::vector<cv::KeyPoint> k(M), ku(M);
    std::vector<float> ur(M), d(M);
    std::vector<MapPoint*> mp(M);
    std::vector<bool> out(M);
    cv::Mat desc(M, mDescriptors.cols, mDescriptors.type());
    for(int j=0;j<M;j++)
    {
        const int i = keep[j];
        k[j]=mvKeys[i]; ku[j]=mvKeysUn[i]; ur[j]=mvuRight[i]; d[j]=mvDepth[i];
        mp[j]=mvpMapPoints[i]; out[j]=mvbOutlier[i];
        mDescriptors.row(i).copyTo(desc.row(j));
    }
    mvKeys=k; mvKeysUn=ku; mvuRight=ur; mvDepth=d; mvpMapPoints=mp; mvbOutlier=out; mDescriptors=desc;
    N = M;
    mBowVec.clear(); mFeatVec.clear();
    for(unsigned int i=0; i<FRAME_GRID_COLS;i++)
        for (unsigned int j=0; j<FRAME_GRID_ROWS;j++)
            mGrid[i][j].clear();
    AssignFeaturesToGrid();
}

// ---------------------------------------------------------------------------------------------
bool Tracking::IsGTDynamic(const cv::KeyPoint &kp) const
{
    if(mGTMask.empty()) return false;
    int x = cvRound(kp.pt.x), y = cvRound(kp.pt.y);
    if(x<0 || y<0 || x>=mGTMask.cols || y>=mGTMask.rows) return false;
    return mGTMask.at<uchar>(y,x) > 0;
}

void Tracking::CountDecisions(const std::vector<bool> &vbDyn, bool ran)
{
    mStats.ran = ran ? 1 : 0;
    mStats.TP = mStats.FP = mStats.FN = mStats.TN = 0;
    for(int i=0;i<mCurrentFrame.N;i++)
    {
        const bool gt = IsGTDynamic(mCurrentFrame.mvKeysUn[i]);
        const bool pr = vbDyn[i];
        if(gt && pr) mStats.TP++; else if(!gt && pr) mStats.FP++;
        else if(gt && !pr) mStats.FN++; else mStats.TN++;
    }
}

void Tracking::PreTrackFilter()
{
    mStats = FrameStats();
    mStats.N = mCurrentFrame.N;
    for(int i=0;i<mCurrentFrame.N;i++) if(mCurrentFrame.mvDepth[i]>0) mStats.nDepth++;
    std::vector<bool> vbDyn(mCurrentFrame.N,false);
    if(mDynCfg.type==1 && !mPrevGray.empty())
    {
        vbDyn = FlowFilter();
        CountDecisions(vbDyn, true);
        bool bAny=false; for(size_t i=0;i<vbDyn.size();i++) bAny = bAny || vbDyn[i];
        if(bAny) mCurrentFrame.RemoveKeypoints(vbDyn);
    }
    else
        CountDecisions(vbDyn, false);   // NONE, GEOM (overwritten in Track if GEOM runs), or first frame
}

// ---------------------------------------------------------------------------------------------
// FLOW: backward pyramidal LK to frame t-1, RANSAC fundamental matrix, point-to-epipolar-line residual.
std::vector<bool> Tracking::FlowFilter()
{
    const int N = mCurrentFrame.N;
    std::vector<bool> vbDyn(N,false);
    if(N<8) return vbDyn;
    std::vector<cv::Point2f> cur(N), prev;
    for(int i=0;i<N;i++) cur[i]=mCurrentFrame.mvKeysUn[i].pt;
    std::vector<uchar> status; std::vector<float> err;
    cv::calcOpticalFlowPyrLK(mImGray, mPrevGray, cur, prev, status, err,
                             cv::Size(mDynCfg.lkWin,mDynCfg.lkWin), mDynCfg.lkLevels-1,
                             cv::TermCriteria(cv::TermCriteria::COUNT+cv::TermCriteria::EPS,30,0.01));
    const float W = (float)mImGray.cols, H = (float)mImGray.rows, b = mDynCfg.border;
    std::vector<int> idx; std::vector<cv::Point2f> pc, pp;
    for(int i=0;i<N;i++)
    {
        if(!status[i] || err[i]>mDynCfg.lkErr) continue;
        const cv::Point2f &a=cur[i], &q=prev[i];
        if(a.x<b || a.y<b || a.x>W-1-b || a.y>H-1-b) continue;
        if(q.x<b || q.y<b || q.x>W-1-b || q.y>H-1-b) continue;
        idx.push_back(i); pc.push_back(a); pp.push_back(q);
    }
    if(idx.size()<15) return vbDyn;
    cv::Mat inl;
    cv::Mat F = cv::findFundamentalMat(pp, pc, cv::FM_RANSAC, mDynCfg.ransacF, 0.99, inl);
    if(F.empty() || F.rows!=3) return vbDyn;
    F.convertTo(F, CV_64F);
    const double *f = F.ptr<double>(0);
    for(size_t j=0;j<idx.size();j++)
    {
        const double x1=pp[j].x, y1=pp[j].y, x2=pc[j].x, y2=pc[j].y;
        // epipolar line in the current image: l = F * x'
        const double l1=f[0]*x1+f[1]*y1+f[2], l2=f[3]*x1+f[4]*y1+f[5], l3=f[6]*x1+f[7]*y1+f[8];
        const double den = std::sqrt(l1*l1+l2*l2);
        if(den<1e-12) continue;
        const double dist = std::fabs(x2*l1+y2*l2+l3)/den;
        if(dist>mDynCfg.tauEpi) vbDyn[idx[j]]=true;
    }
    if(mDynCfg.rf>0.f)
    {
        std::vector<bool> vbR = vbDyn;
        for(int i=0;i<N;i++) if(vbDyn[i])
        {
            std::vector<size_t> near = mCurrentFrame.GetFeaturesInArea(cur[i].x,cur[i].y,mDynCfg.rf);
            for(size_t k=0;k<near.size();k++) vbR[near[k]]=true;
        }
        vbDyn = vbR;
    }
    return vbDyn;
}

// ---------------------------------------------------------------------------------------------
// GEOM: multi-view depth consistency against the Ko keyframes that share most map points.
std::vector<bool> Tracking::GeomFilter()
{
    const int N = mCurrentFrame.N;
    std::vector<bool> vbDyn(N,false);
    // 1. overlapping keyframes from the preliminary matches
    std::map<KeyFrame*,int> counter;
    for(int i=0;i<N;i++)
    {
        MapPoint* pMP = mCurrentFrame.mvpMapPoints[i];
        if(!pMP || mCurrentFrame.mvbOutlier[i] || pMP->isBad()) continue;
        const std::map<KeyFrame*,size_t> obs = pMP->GetObservations();
        for(std::map<KeyFrame*,size_t>::const_iterator it=obs.begin(); it!=obs.end(); ++it) counter[it->first]++;
    }
    std::vector<std::pair<int,KeyFrame*> > v;
    for(std::map<KeyFrame*,int>::iterator it=counter.begin(); it!=counter.end(); ++it)
        if(!it->first->isBad()) v.push_back(std::make_pair(it->second,it->first));
    std::sort(v.begin(), v.end(), [](const std::pair<int,KeyFrame*>&a, const std::pair<int,KeyFrame*>&b){ return a.first>b.first; });
    if((int)v.size()>mDynCfg.Ko) v.resize(mDynCfg.Ko);
    if(v.empty()) return vbDyn;

    const cv::Mat Tcw = mCurrentFrame.mTcw;
    const cv::Mat Rcw = Tcw.rowRange(0,3).colRange(0,3), tcw = Tcw.rowRange(0,3).col(3);
    const cv::Mat Ow = -Rcw.t()*tcw;
    const float cosMax = std::cos(mDynCfg.alphaMaxDeg*(float)CV_PI/180.f);
    const float W = (float)mImGray.cols, H = (float)mImGray.rows;
    const bool rgbd = (mSensor==System::RGBD) && !mCurDepth.empty();
    std::vector<cv::Point2f> seeds;
    for(size_t kk=0; kk<v.size(); kk++)
    {
        KeyFrame* pKF = v[kk].second;
        const cv::Mat Okf = pKF->GetCameraCenter();
        for(int i=0;i<pKF->N;i++)
        {
            if(pKF->mvDepth[i]<=0) continue;
            cv::Mat X = pKF->UnprojectStereo(i);
            if(X.empty()) continue;
            cv::Mat Xc = Rcw*X + tcw;
            const float z = Xc.at<float>(2);
            if(z<=0.05f) continue;
            const float u = Frame::fx*Xc.at<float>(0)/z + Frame::cx, vv = Frame::fy*Xc.at<float>(1)/z + Frame::cy;
            if(u<0 || vv<0 || u>W-1 || vv>H-1) continue;
            cv::Mat r1 = X-Okf, r2 = X-Ow;
            const float c = (float)(r1.dot(r2)/(cv::norm(r1)*cv::norm(r2)+1e-12));
            if(c<cosMax) continue;               // parallax > alpha_max: possible self-occlusion
            float zm = -1.f; cv::Point2f sp(u,vv);
            // Measured depth z' at x': the FARTHEST valid depth in a small neighbourhood, so that a projection
            // that lands on the foreground side of a depth edge (sub-pixel pose/rounding error) is not a seed.
            if(rgbd)
            {
                const int cu=cvRound(u), cv_=cvRound(vv), r=mDynCfg.patchR;
                for(int yy=std::max(0,cv_-r); yy<=std::min(mCurDepth.rows-1,cv_+r); yy++)
                    for(int xx=std::max(0,cu-r); xx<=std::min(mCurDepth.cols-1,cu+r); xx++)
                        zm = std::max(zm, mCurDepth.at<float>(yy,xx));
            }
            else
            {
                std::vector<size_t> near = mCurrentFrame.GetFeaturesInArea(u,vv,mDynCfg.stereoAssoc);
                float best=1e9f;
                for(size_t q=0;q<near.size();q++)
                {
                    const float dq = mCurrentFrame.mvDepth[near[q]];
                    if(dq<=0) continue;
                    zm = std::max(zm, dq);
                    const cv::KeyPoint &kp = mCurrentFrame.mvKeysUn[near[q]];
                    const float dd = (kp.pt.x-u)*(kp.pt.x-u)+(kp.pt.y-vv)*(kp.pt.y-vv);
                    if(dd<best){ best=dd; sp=kp.pt; }
                }
            }
            if(zm<=0) continue;
            if(z - zm > mDynCfg.tauZ) seeds.push_back(sp);  // something closer than the static surface
        }
    }
    mStats.nSeeds = (int)seeds.size();
    if(seeds.empty()) return vbDyn;

    if(rgbd)
    {
        // region growing on the depth image from each seed (4-neighbourhood, depth step < tauG),
        // limited to growMax pixels (Chebyshev distance) from the seed
        cv::Mat mask = cv::Mat::zeros(mCurDepth.size(), CV_8U);
        const int R = mDynCfg.growMax;
        for(size_t s=0;s<seeds.size();s++)
        {
            const int sx=cvRound(seeds[s].x), sy=cvRound(seeds[s].y);
            if(mask.at<uchar>(sy,sx)) continue;
            if(mCurDepth.at<float>(sy,sx)<=0) continue;
            std::deque<cv::Point> q; q.push_back(cv::Point(sx,sy)); mask.at<uchar>(sy,sx)=255;
            while(!q.empty())
            {
                cv::Point p=q.front(); q.pop_front();
                const float dp = mCurDepth.at<float>(p.y,p.x);
                const int nx[4]={p.x+1,p.x-1,p.x,p.x}, ny[4]={p.y,p.y,p.y+1,p.y-1};
                for(int k=0;k<4;k++)
                {
                    const int x=nx[k], y=ny[k];
                    if(x<0||y<0||x>=mask.cols||y>=mask.rows) continue;
                    if(std::abs(x-sx)>R || std::abs(y-sy)>R) continue;
                    if(mask.at<uchar>(y,x)) continue;
                    const float dn = mCurDepth.at<float>(y,x);
                    if(dn<=0 || std::fabs(dn-dp)>=mDynCfg.tauG) continue;
                    mask.at<uchar>(y,x)=255; q.push_back(cv::Point(x,y));
                }
            }
        }
        for(int i=0;i<N;i++)
        {
            const cv::Point2f &pt = mCurrentFrame.mvKeysUn[i].pt;
            const int x=cvRound(pt.x), y=cvRound(pt.y);
            if(x>=0&&y>=0&&x<mask.cols&&y<mask.rows && mask.at<uchar>(y,x)) vbDyn[i]=true;
        }
    }
    else
    {
        for(size_t s=0;s<seeds.size();s++)
        {
            std::vector<size_t> near = mCurrentFrame.GetFeaturesInArea(seeds[s].x,seeds[s].y,mDynCfg.rg);
            for(size_t k=0;k<near.size();k++) vbDyn[near[k]]=true;
        }
    }
    return vbDyn;
}

// ---------------------------------------------------------------------------------------------
void Tracking::OpenLog(const std::string &path)
{
    mpLogFile = fopen(path.c_str(),"w");
    if(mpLogFile) fprintf(mpLogFile,"frame,t,state,N,ran,TP,FP,FN,TN,nDepth,nSeeds,nKept,nInl,nInlStatic,nKF,nMP,resets\n");
}
void Tracking::CloseLog(){ if(mpLogFile){ fclose(mpLogFile); mpLogFile=nullptr; } }

void Tracking::LogFrame()
{
    if(!mpLogFile) return;
    int nInl=0, nInlS=0;
    if(mState==OK)
        for(int i=0;i<mCurrentFrame.N;i++)
        {
            MapPoint* pMP = mCurrentFrame.mvpMapPoints[i];
            if(pMP && !mCurrentFrame.mvbOutlier[i])
            {
                nInl++;
                if(!IsGTDynamic(mCurrentFrame.mvKeysUn[i])) nInlS++;
            }
        }
    fprintf(mpLogFile,"%lu,%.6f,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d\n",
            mCurrentFrame.mnId, mCurrentFrame.mTimeStamp, (int)mState, mStats.N, mStats.ran,
            mStats.TP, mStats.FP, mStats.FN, mStats.TN, mStats.nDepth, mStats.nSeeds, mCurrentFrame.N,
            nInl, nInlS, (int)mpMap->KeyFramesInMap(), (int)mpMap->MapPointsInMap(), mnResets);
}

} // namespace ORB_SLAM2
