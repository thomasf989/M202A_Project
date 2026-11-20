#import "OfflineLocalizer.h"
#include <TargetConditionals.h>
#if TARGET_OS_SIMULATOR
@implementation OfflineLocalizer
- (instancetype)initWithDatabase:(NSString *)path error:(NSError **)error { return [super init]; }
- (NSDictionary *)localizeJPEG:(NSData *)jpeg intrinsics:(NSArray<NSNumber *> *)intrinsics {
    return @{ @"success":@(false), @"reason":@"Offline localization requires a physical iPhone in this build." };
}
@end
#else
#include <opencv2/opencv.hpp>
#include <set>
#include <numeric>
#include <algorithm>

using namespace cv;
struct Reference { Mat desc, xyz, valid; };
struct Support { std::vector<int> indices; double median = 1e9; };
static Support support(const std::vector<Point3f>& xyz, const std::vector<Point2f>& xy,
                       const std::vector<int>& qids, const Mat& K, const Mat& r, const Mat& t) {
    std::vector<Point2f> projected; projectPoints(xyz, r, t, K, noArray(), projected);
    Mat R; Rodrigues(r, R);
    std::vector<std::pair<double,int>> ordered;
    for (int i=0;i<(int)xyz.size();i++) {
        auto p=xyz[i]; double z=R.at<double>(2,0)*p.x+R.at<double>(2,1)*p.y+R.at<double>(2,2)*p.z+t.at<double>(2);
        double e=norm(projected[i]-xy[i]); if (z>.05 && e<4.) ordered.emplace_back(e,i);
    }
    std::sort(ordered.begin(),ordered.end()); std::set<int> seen; Support s;
    std::vector<double> errors;
    for (auto [e,i]:ordered) if(seen.insert(qids[i]).second) {s.indices.push_back(i);errors.push_back(e);}
    if (!errors.empty()) s.median=errors[errors.size()/2]; return s;
}

@implementation OfflineLocalizer {
    Mat centers, idf, vectors;
    std::vector<Reference> refs;
    int nfeatures;
}
- (instancetype)initWithDatabase:(NSString *)path error:(NSError **)error {
    if ((self=[super init])) {
        try {
            FileStorage fs(path.UTF8String,FileStorage::READ);
            if(!fs.isOpened()) CV_Error(Error::StsError,"Missing offline room database");
            fs["centers"] >> centers; fs["idf"] >> idf; fs["vectors"] >> vectors;
            fs["nfeatures"] >> nfeatures; int count=0; fs["count"] >> count;
            if(count<4 || centers.cols!=128 || vectors.rows!=count) CV_Error(Error::StsError,"Invalid room database");
            for(int i=0;i<count;i++) { Reference f; auto k=std::to_string(i);
                fs["desc_"+k] >> f.desc; fs["xyz_"+k] >> f.xyz; fs["valid_"+k] >> f.valid;
                if(f.desc.cols!=128 || f.xyz.cols!=3 || f.desc.rows!=f.xyz.rows || f.valid.total()!=f.desc.rows)
                    CV_Error(Error::StsError,"Invalid reference geometry");
                refs.push_back(f);
            }
            setNumThreads(2);
        } catch(const std::exception& e) {
            if(error) *error=[NSError errorWithDomain:@"RoomLocalizer" code:1 userInfo:@{NSLocalizedDescriptionKey:@(e.what())}];
            return nil;
        }
    } return self;
}
- (NSDictionary *)localizeJPEG:(NSData *)jpeg intrinsics:(NSArray<NSNumber *> *)intrinsics {
    try {
        if(intrinsics.count!=9) CV_Error(Error::StsError,"Missing camera calibration");
        Mat K(3,3,CV_64F); for(int i=0;i<9;i++) K.at<double>(i/3,i%3)=intrinsics[i].doubleValue;
        Mat bytes(1,(int)jpeg.length,CV_8U,(void*)jpeg.bytes), gray=imdecode(bytes,IMREAD_GRAYSCALE);
        if(gray.empty() || gray.total()>4096*4096 || !checkRange(K) || K.at<double>(0,0)<=0 || K.at<double>(1,1)<=0)
            CV_Error(Error::StsError,"Invalid image/calibration");
        Mat mask=gray>8; erode(mask,mask,Mat::ones(9,9,CV_8U));
        std::vector<KeyPoint> keypoints; Mat desc;
        SIFT::create(nfeatures,3,.025)->detectAndCompute(gray,mask,keypoints,desc);
        if(desc.rows<24) return @{ @"success":@(false), @"reason":@"Not enough visual detail. Aim at textured furniture or corners." };
        for(int i=0;i<desc.rows;i++) { float total=(float)sum(desc.row(i))[0]; desc.row(i)/=std::max(total,1e-8f); }
        sqrt(desc,desc);
        BFMatcher matcher(NORM_L2); std::vector<DMatch> assigned; matcher.match(desc,centers,assigned);
        Mat hist=Mat::zeros(1,centers.rows,CV_32F);
        for(auto m:assigned) hist.at<float>(0,m.trainIdx)+=1.f;
        hist/=desc.rows; sqrt(hist,hist); multiply(hist,idf,hist); normalize(hist,hist);
        std::vector<int> ranking(refs.size()); std::iota(ranking.begin(),ranking.end(),0);
        std::sort(ranking.begin(),ranking.end(),[&](int a,int b){return vectors.row(a).dot(hist)>vectors.row(b).dot(hist);});
        std::vector<Point3f> xyz; std::vector<Point2f> xy; std::vector<int> qids;
        std::vector<std::vector<int>> groups;
        for(int ri:std::vector<int>(ranking.begin(),ranking.begin()+std::min<size_t>(8,ranking.size()))) {
            auto& f=refs[ri]; std::vector<std::vector<DMatch>> forward; std::vector<DMatch> backward;
            if(f.desc.rows<2) continue;
            matcher.knnMatch(desc,f.desc,forward,2); matcher.match(f.desc,desc,backward);
            std::vector<int> group;
            for(auto& pair:forward) if(pair.size()==2) {
                auto a=pair[0];
                if(a.distance>=.75*pair[1].distance || backward[a.trainIdx].trainIdx!=a.queryIdx || !f.valid.at<uchar>(a.trainIdx)) continue;
                group.push_back((int)xyz.size()); const float* p=f.xyz.ptr<float>(a.trainIdx);
                xyz.emplace_back(p[0],p[1],p[2]); xy.push_back(keypoints[a.queryIdx].pt); qids.push_back(a.queryIdx);
            } groups.push_back(group);
        }
        std::vector<int> all(xyz.size()); std::iota(all.begin(),all.end(),0); groups.insert(groups.begin(),all);
        Mat bestR,bestT; Support best;
        for(auto& g:groups) {
            if(g.size()<12) continue;
            std::vector<Point3f> x; std::vector<Point2f> u; for(int i:g) {x.push_back(xyz[i]);u.push_back(xy[i]);}
            Mat r,t,inliers; setRNGSeed(42);
            if(!solvePnPRansac(x,u,K,noArray(),r,t,false,3000,4.,.999,inliers,SOLVEPNP_EPNP)) continue;
            Support s=support(xyz,xy,qids,K,r,t);
            if(s.indices.size()>best.indices.size() || (s.indices.size()==best.indices.size() && s.median<best.median)) {best=s;bestR=r;bestT=t;}
        }
        for(int pass=0;pass<2 && best.indices.size()>=12;pass++) {
            std::vector<Point3f> x; std::vector<Point2f> u; for(int i:best.indices) {x.push_back(xyz[i]);u.push_back(xy[i]);}
            solvePnPRefineLM(x,u,K,noArray(),bestR,bestT); best=support(xyz,xy,qids,K,bestR,bestT);
        }
        std::set<int> unique(qids.begin(),qids.end()), cells;
        for(int i:best.indices) cells.insert((int)(xy[i].y/(gray.rows/4.))*4+(int)(xy[i].x/(gray.cols/4.)));
        if(best.indices.size()<12 || best.indices.size()<.12*unique.size() || cells.size()<4)
            return @{ @"success":@(false), @"reason":@"No reliable pose. Try a wider, sharper view of the scanned room." };
        Mat points((int)best.indices.size(),3,CV_64F);
        for(int j=0;j<points.rows;j++) for(int c=0;c<3;c++) points.at<double>(j,c)=(c==0?xyz[best.indices[j]].x:c==1?xyz[best.indices[j]].y:xyz[best.indices[j]].z);
        Mat mean; reduce(points,mean,0,REDUCE_AVG); for(int j=0;j<points.rows;j++) points.row(j)-=mean;
        Mat singular; SVD::compute(points,singular);
        if(singular.at<double>(1)<.03) return @{ @"success":@(false), @"reason":@"Insufficient 3D spread." };
        Mat R; Rodrigues(bestR,R); Mat rotation=R.t(), position=-rotation*bestT;
        NSMutableArray* pose=[NSMutableArray array];
        for(int row=0;row<4;row++) for(int col=0;col<4;col++) {
            double value=row==3?(col==3?1:0):(col==3?position.at<double>(row):rotation.at<double>(row,col));
            [pose addObject:@(value)];
        }
        return @{ @"success":@(true), @"pose":pose, @"inliers":@(best.indices.size()), @"reprojection":@(best.median) };
    } catch(const std::exception& e) {
        return @{ @"success":@(false), @"reason":@(e.what()) };
    }
}
@end

#endif
