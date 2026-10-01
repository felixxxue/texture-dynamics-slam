// run_seq: run ORB-SLAM2 (stereo or RGB-D) with a dynamic-point filter on one rendered sequence.
// usage: run_seq <vocab.bin> <settings.yaml> <seqdir> <stereo|rgbd> <none|flow|geom> <outdir> [key=value ...]
#include "System.h"
#include "Tracking.h"
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include <iostream>
#include <fstream>
#include <chrono>
#include <thread>
#include <cstdio>
using namespace std;
int main(int argc, char** argv)
{
    if(argc<7){ cerr << "usage: run_seq vocab settings seqdir stereo|rgbd none|flow|geom outdir [key=value]" << endl; return 1; }
    string voc=argv[1], settings=argv[2], seq=argv[3], mode=argv[4], filt=argv[5], out=argv[6];
    ORB_SLAM2::DynFilterConfig cfg;
    cfg.type = filt=="flow" ? 1 : (filt=="geom" ? 2 : 0);
    int realtime=1, maskDilate=2; double fps=30.0;
    for(int a=7;a<argc;a++)
    {
        string kv=argv[a]; size_t e=kv.find('='); if(e==string::npos) continue;
        string k=kv.substr(0,e); double v=atof(kv.substr(e+1).c_str());
        if(k=="realtime") realtime=(int)v; else if(k=="delta") maskDilate=(int)v; else if(k=="fps") fps=v;
        else if(k=="lkErr") cfg.lkErr=(float)v; else if(k=="tauEpi") cfg.tauEpi=(float)v; else if(k=="rf") cfg.rf=(float)v;
        else if(k=="tauZ") cfg.tauZ=(float)v; else if(k=="tauG") cfg.tauG=(float)v; else if(k=="rg") cfg.rg=(float)v;
        else if(k=="Ko") cfg.Ko=(int)v; else if(k=="growMax") cfg.growMax=(int)v; else if(k=="ransacF") cfg.ransacF=(float)v;
    }
    vector<double> times; { ifstream f(seq+"/times.txt"); double t; while(f>>t) times.push_back(t); }
    if(times.empty()){ cerr << "no times.txt" << endl; return 2; }
    const bool stereo = (mode=="stereo");
    ORB_SLAM2::System SLAM(voc, settings, stereo ? ORB_SLAM2::System::STEREO : ORB_SLAM2::System::RGBD, false);
    ORB_SLAM2::Tracking* tr = SLAM.GetTracker();
    tr->SetDynFilter(cfg);
    tr->OpenLog(out+"/frames.csv");
    cv::Mat ker = cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(2*maskDilate+1,2*maskDilate+1));
    const double dt = 1.0/fps;
    auto t0 = std::chrono::steady_clock::now();
    double ttrack_sum=0;
    char buf[512];
    for(size_t i=0;i<times.size();i++)
    {
        sprintf(buf,"%s/left/%06d.png",seq.c_str(),(int)i); cv::Mat L=cv::imread(buf,cv::IMREAD_UNCHANGED);
        sprintf(buf,"%s/mask/%06d.png",seq.c_str(),(int)i); cv::Mat M=cv::imread(buf,cv::IMREAD_GRAYSCALE);
        if(L.empty()){ cerr << "missing " << buf << endl; return 3; }
        if(!M.empty() && maskDilate>0) cv::dilate(M,M,ker);
        tr->SetGTMask(M);
        auto a = std::chrono::steady_clock::now();
        if(stereo){ sprintf(buf,"%s/right/%06d.png",seq.c_str(),(int)i); cv::Mat R=cv::imread(buf,cv::IMREAD_UNCHANGED); SLAM.TrackStereo(L,R,times[i]); }
        else { sprintf(buf,"%s/depth/%06d.png",seq.c_str(),(int)i); cv::Mat D=cv::imread(buf,cv::IMREAD_UNCHANGED); SLAM.TrackRGBD(L,D,times[i]); }
        auto b = std::chrono::steady_clock::now();
        ttrack_sum += std::chrono::duration<double>(b-a).count();
        if(realtime)
            std::this_thread::sleep_until(t0 + std::chrono::duration_cast<std::chrono::steady_clock::duration>(std::chrono::duration<double>((i+1)*dt)));
    }
    SLAM.Shutdown();
    tr->CloseLog();
    SLAM.SaveTrajectoryTUM(out+"/traj.txt");
    FILE* f=fopen((out+"/run_info.txt").c_str(),"w");
    if(f){ fprintf(f,"frames=%d\nmean_track_ms=%.3f\nresets=%d\n",(int)times.size(),1000.0*ttrack_sum/times.size(),tr->mnResets); fclose(f); }
    return 0;
}
