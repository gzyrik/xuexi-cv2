#include <opencv2/opencv.hpp>
#include <cmath>
static double nuclear_norm(const cv::Mat& M){
    return std::sqrt(cv::sum((M*M.t()).diag())[0]);//核范数
}
int RobustPCA_IALM(const cv::Mat& D, cv::Mat& A, cv::Mat& E,
    double r = -1, double tol = 1e-4, const int maxIter = 1000)
{
    const bool f64 = (D.type() == CV_64FC1);
    const double sv_n[6]={0.02, 0.06, 0.26, 0.28, 0.34, 0.38};//sv/n 比值
    const auto dims = D.dims;
    const auto size = D.size;
    if (r  <= 0) r = 1/std::sqrt(std::max(size[0], size[1]));
    const int n = size[1], sn = static_cast<int>(std::round(0.05*n));
    const int sv_max = static_cast<int>(n*sv_n[std::min(5, (n-1)/100)]);

    const auto norm_two = cv::norm(D, cv::NORM_L2);
    const auto norm_inf = cv::norm(D, cv::NORM_INF) / r;
    const auto dual_norm = std::max(norm_two, norm_inf);


    cv::Mat U, S, V;
    auto Y = D/dual_norm;
    auto u = 1.25/norm_two;
    const auto u_max = u * 1e7;
    const auto s = 1.5;
    int sv = 10;
    A = cv::Mat::zeros(D.size(), D.type());
    int i = 0;
    while (1) {
        //更新 E
        const auto _u = 1/u, r_u=r/u;
        const auto Y_u = Y /u;
        E = D - A + Y_u;
        if (f64){
            for (auto p = E.ptr<double>(), pEnd=p+E.total(); p<pEnd; ++p)
                *p = std::max(*p - r_u, 0.0) + std::min(*p + r_u, 0.0);
        }
        else {
            for (auto p = E.ptr<float>(), pEnd=p+E.total(); p<pEnd; ++p)
                *p = std::max(*p - (float)r_u, 0.0f) + std::min(*p + (float)r_u, 0.0f);
        }
        //更新 A, U*S*V == D-A+Y/u
        A = D-E+Y_u;
        int total;
        if (sv < sv_max){
            cv::SVD::compute(A, S, U, V);//choosvd
            total = sv;
        }
        else {
            cv::SVD::compute(A, S, U, V);
            total = (int)S.total();
        }
        int svp = 0;
        if (f64){
            for (auto p = S.ptr<double>(); svp<total && p[svp] > _u; ++svp)
                p[svp] -= _u;
        }
        else {
            for (auto p = S.ptr<float>(); svp<total && p[svp] > _u; ++svp)
                p[svp] -= (float)_u;
        }

        if (svp < sv)
            sv = std::min(svp+1, n);
        else
            sv = std::min(svp+sn, n);

        U = U.colRange(0, svp);
        S = S.rowRange(0, svp);
        V = V.rowRange(0, svp);
        A = U*cv::Mat::diag(S)*V;//U*(S-1/u)*V

        const auto Z = D - A - E;
        Y = Y + u * Z; //更新,乘子Y
        u = std::min(u * s, u_max); //更新,惩罚因子u
        const auto e = cv::norm(Z, cv::NORM_L1);///norm_two;

        if (i % 10 == 0 || e < tol || i >= maxIter) {
            auto minA = nuclear_norm(A);//核范数
            auto minE = cv::norm(E,cv::NORM_L1);//绝对值和范数
            std::cout << std::endl << i<<","<<svp
                << "\t|A|*="  << minA 
                << "\t|E|_1=" << minE
                << "\tmin="   << (minA + r*minE)
                << "\terr="   << e;
        }
        if ( e < tol || i >= maxIter)//限制最大的迭代次数
            break;
        ++i;
    }
    return i;
}
void test()
{
    const auto M = 180;
    const auto N = 2304;
    const auto rank = 5;
    const auto card = 0.20;
    cv::RNG rng;
    const int type = CV_64FC1;
    cv::Mat R(rank, N, type);
    rng.fill(R, cv::RNG::UNIFORM, 0.0,1.0);

    //低秩矩阵
    cv::Mat A0(M, N, type);
    for(int i=0;i<M;++i)
        R.row(rng.uniform(0, rank)).copyTo(A0.row(i));

    //稀疏矩阵
    cv::Mat E0(M,N, type);
    E0.forEach<double>([&](double& e, const int position[]) {
        e = (rng.uniform(0.0,1.0) < card);
    });

    cv::Mat A1, E1;
    double r=1/std::sqrt(std::max(M,N));
    RobustPCA_IALM(A0 + E0, A1, E1, r);
    auto minA = nuclear_norm(A0);//核范数
    auto minE = cv::norm(E0,cv::NORM_L1);//绝对值和范数
    std::cout << std::endl
        << "\t|A|*="  << minA 
        << "\t|E|_1=" << minE
        << "\tmin="   << (minA + r*minE);

}
static std::string _title;
void main(int argc, char* argv[])
{
    //return test();
    cv::VideoCapture movie("res\\RobustPCA_video_demo.avi");
    if (!movie.isOpened()) {
        exit(-1);
    }
    auto frames= (int)movie.get(cv::CAP_PROP_FRAME_COUNT);
    auto fps   = (int)movie.get(cv::CAP_PROP_FPS);
    auto width = (int)movie.get(cv::CAP_PROP_FRAME_WIDTH);
    auto height= (int)movie.get(cv::CAP_PROP_FRAME_HEIGHT);

    {
        std::ostringstream oss;
        oss << width << "x" << height << " FPS:" << fps << " Frames:" << frames;
        _title = oss.str();
        std::cout << _title << std::endl;
    }

    cv::Mat D(frames, width*height, CV_64FC1);
    cv::Mat frame, gray;
    for (int i=0;i<frames; ++i) {
        movie >> frame;
        if (CV_8UC1 != frame.type()){
            cv::cvtColor(frame, gray, cv::COLOR_BGR2GRAY);
            frame = gray;
        }
        frame.reshape(1, 1).convertTo(D.row(i), CV_64FC1,1/255.0);
    }
    cv::Mat A, E;
    RobustPCA_IALM(D, A, E);
    std::ofstream fsA,fsE;
    {
        std::ostringstream oss;
        oss << "A_" << width << "x" << height <<".y";
        fsA.open(oss.str(), std::ofstream::binary);
        if (!fsA.is_open()){
            std::cerr << std::endl << oss.str() << "open failed";
            exit(-1);
        }
    }
    {
        std::ostringstream oss;
        oss << "E_" << width << "x" << height <<".y";
        fsE.open(oss.str(), std::ofstream::binary);
        if (!fsE.is_open()){
            std::cerr << std::endl << oss.str() << "open failed";
            exit(-1);
        }
    }
    for (int i=0;i<frames;++i) {
        A.row(i).convertTo(frame, CV_8UC1, 255);
        frame.reshape(1, height);
        fsA.write(frame.ptr<char>(), width*height);

        E.row(i).convertTo(frame, CV_8UC1, 255);
        frame.reshape(1, height);
        fsE.write(frame.ptr<char>(), width*height);
    }
}
