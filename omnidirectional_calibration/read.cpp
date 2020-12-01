#define _SILENCE_EXPERIMENTAL_FILESYSTEM_DEPRECATION_WARNING
#include <opencv2/opencv.hpp>
#include "opencv2/core.hpp"
#include "opencv2/core/affine.hpp"
#include <iostream>
#include <vector>
#include <string>
#include <algorithm>
#include <experimental/filesystem>
#include <opencv2/ccalib/omnidir.hpp>
#include <math.h>

namespace fs = std::experimental::filesystem;

std::vector<std::string> fileList(std::string dir, std::string ext = "")
{
    //対象のディレクトリ直下のファイルのリスト取得
    std::vector<std::string> flist;
    for (auto ent : fs::directory_iterator(dir)) {
        if (!fs::is_directory(ent)) {
            std::string path = ent.path().string();
            if (ext != "") {
                auto pos = path.rfind(ext);
                if (pos == std::string::npos)
                    continue;
                if (pos + ext.length() != path.length())
                    continue;
            }
            flist.push_back(path);
        }
    }
    std::sort(flist.begin(), flist.end());
    return flist;
}

int main() {
    std::string imgDir = "./image/Lcut4";
    std::string imgDir2 = "./image/Rcut4";
    std::string imgPath;
    cv::Mat img2;
    cv::Mat imgP;
    cv::Mat tvec2T;
    std::vector<std::string> imgList = fileList(imgDir, ".png");
    std::vector<std::string> imgList2 = fileList(imgDir2, ".png");
    cv::Mat K1, K2, xi1, xi2, D1, D2;
    std::vector<cv::Mat> rvecsL, tvecsL;
    cv::Mat rvec, tvec, idx;
    std::vector<std::vector<cv::Point2f>> imgPoints;
    std::vector<std::vector<cv::Point2f>> imgPoints2;
    std::vector<std::vector<cv::Point3f>> objPoints;
    std::vector<cv::Point2f> corners;
    std::vector<cv::Point2f> corners2;
    std::vector<cv::Point3f> obj;
    cv::FileStorage fs("stereoCalibrate3.xml", cv::FileStorage::READ);
    fs["K1"] >> K1;
    fs["xi1"] >> xi1;
    fs["D1"] >> D1;
    fs["K2"] >> K2;
    fs["xi2"] >> xi2;
    fs["D2"] >> D2;
    fs["rvec"] >> rvec;
    fs["tvec"] >> tvec;
    fs["rvecsL"] >> rvecsL;
    fs["tvecsL"] >> tvecsL;
    fs["idx"] >> idx;

    for (int i = 0; i < 20; i++) {
        std::ostringstream oss;
        oss << i;
        fs["imgPoints_" + oss.str()] >> corners;
        imgPoints.push_back(corners);
        fs["imgPoints2_" + oss.str()] >> corners2;
        imgPoints2.push_back(corners2);
        fs["objPoints_" + oss.str()] >> obj;
        objPoints.push_back(obj);

    }
    std::cout << "imgSize" << imgPoints[0] << std::endl;
    
    std::cout << "K1" << K1 << std::endl;
    std::cout << "K1" << D1 << std::endl;
    
    double* dxi = xi1.ptr<double>(0);
    double* dxi2 = xi2.ptr<double>(0);
    int* idx2 = idx.ptr<int>(0);
    cv::Mat rvecL1, tvecL1, yaco,rvecsL_R,rvecsL2_R, rvec_R,tvec2,rvec2;

    cv::Mat img;

    cv::Mat tvec_t;
    
    //cv::Mat img = cv::imread("./image/cutL3/cut16.png");
    //cv::Mat img2 = cv::imread("./image/cutR3/cut16.png");

    //cv::Mat img2 = cv::imread("./image/cutR3/cut33.png");
    std::vector<cv::Point2f > obj_origin_img;
    std::vector<cv::Point2f > obj_origin_imgc;
    std::vector<cv::Point2f > obj_origin_imgx;
    std::vector<cv::Point2f > obj_origin_imgy;
    std::vector<cv::Point2f > obj_origin_imgz;
    cv::Point2f error;
    
    double esum = 0;
    double eave = 0;
    
    //cv::solvePnP(objPoints[0], imgPoints[5], K1, D1, rvecL1, tvecL1, 0);
    std::cout << "rvecsL" << rvecsL[0] << std::endl;
    std::cout << "rvec" << rvec << std::endl;
    std::cout << "tvecsL1" << tvecsL[0] << std::endl;
    std::cout << "tvec" << tvec << std::endl;
    std::cout << "xi2" << xi2 << std::endl;
    std::cout << "idx: " << idx2[1] << std::endl;
    
    std::vector<std::vector<cv::Point3f>> objPointsc = objPoints;
    std::vector<std::vector<cv::Point3f>> objPointsx = objPoints;
    std::vector<std::vector<cv::Point3f>> objPointsy = objPoints;
    std::vector<std::vector<cv::Point3f>> objPointsz = objPoints;
    objPointsx[0][0].x += 389;
    objPointsy[0][0].y += 100;
    objPointsz[0][0].z += 100;
    
    img = cv::imread("./image/Lcut4/cut02.png");
    cv::omnidir::projectPoints(objPoints[0], obj_origin_img,rvecsL[0],tvecsL[0],K1,dxi[0],D1,yaco);
    cv::circle(img, obj_origin_img[0], 20, { 255,0,0 }, -1);
    cv::omnidir::projectPoints(objPointsx[0], obj_origin_imgx, rvecsL[0], tvecsL[0], K1, dxi[0], D1, yaco);
    cv::line(img, obj_origin_img[0], obj_origin_imgx[0], { 255,255,0 }, 5,  -1);
    cv::omnidir::projectPoints(objPointsy[0], obj_origin_imgy, rvecsL[0], tvecsL[0], K1, dxi[0], D1, yaco);
    cv::line(img, obj_origin_img[0], obj_origin_imgy[0], { 0,255,0 }, 5, -1);
    cv::omnidir::projectPoints(objPointsz[0], obj_origin_imgz, rvecsL[0], tvecsL[0], K1, dxi[0], D1, yaco);
    cv::line(img, obj_origin_img[0], obj_origin_imgz[0], { 0,0,255 }, 5, -1);
    std::cout << "obj_origin_img[0]" << objPoints[0] << std::endl;
    cv::omnidir::projectPoints(objPoints[0], obj_origin_img, -rvec, -tvec, K1, dxi[0], D1, yaco);
    std::cout << "obj_origin_img[0]" << obj_origin_img[0] << std::endl;
    cv::circle(img, obj_origin_img[0], 20, { 255,0,0 }, -1);
    cv::imwrite("./result/L.png", img);

    objPointsx[0][0].z += 100;
    objPointsy[0][0].z += 100;
    objPointsz[0][0].z += 100;
    objPointsc[0][0].z += 100;
    cv::Mat Rn = (cv::Mat_<double>(1, 3) << 0, 0, 0);
    cv::Mat Tn = (cv::Mat_<double>(1, 3) << 0, 0, 0);
    imgP = cv::imread("./image/Lcut4/cut02.png");
    cv::omnidir::projectPoints(objPointsc[0], obj_origin_imgc, Rn, Tn, K1, dxi[0], D1, yaco);
    cv::omnidir::projectPoints(objPointsx[0], obj_origin_imgx, Rn, Tn, K1, dxi[0], D1, yaco);
    cv::omnidir::projectPoints(objPointsy[0], obj_origin_imgy, Rn, Tn, K1, dxi[0], D1, yaco);
    cv::omnidir::projectPoints(objPointsz[0], obj_origin_imgz, Rn, Tn, K1, dxi[0], D1, yaco);
    cv::line(imgP, obj_origin_imgc[0], obj_origin_imgx[0], { 255,255,0 }, 5, -1);
    cv::line(imgP, obj_origin_imgc[0], obj_origin_imgy[0], { 0,255,0 }, 5, -1);
    cv::line(imgP, obj_origin_imgc[0], obj_origin_imgz[0], { 0,0,255 }, 5, -1);
    cv::imwrite("./result/C.png", imgP);
    
    //std::cout << "onj" << obj_origin_img << std::endl;
    //std::cout << "onj" << objPoints[0] << std::endl;
    std::cout << "rvecsL_R:" << rvecsL_R << std::endl;
    std::cout << "rvec_R" << rvec_R << std::endl;
    std::cout << "rvec2" << rvec2 << std::endl;
    std::cout << "xi" << dxi[0] << std::endl;
    for (int i = 0; i < 1; i++) {
        imgPath = imgList2[idx2[i]];

        
        cv::Rodrigues(rvecsL[i], rvecsL_R);
        cv::Rodrigues(rvec, rvec_R);
        rvecsL2_R = rvec_R * rvecsL_R;
        tvec_t = rvec_R * tvecsL[i];
        tvec2 = tvec_t + tvec;
        cv::Rodrigues(rvecsL2_R, rvec2);
        std::cout << "tvec_t: " << tvec_t << std::endl;
        std::cout << "idx: " << idx2[i] << std::endl;
        std::cout << "path: " << imgPath << std::endl;
        std::cout << "rvec2: " << rvec2 << std::endl;
        std::cout << "tvec2: " << tvec2 << std::endl;
        std::string substr = imgPath.substr(13);

        cv::omnidir::projectPoints(objPoints[0], obj_origin_img, rvec2, tvec2, K2, dxi2[0], D2, yaco);

        img2 = cv::imread(imgPath);
        for (int j = 0; j < objPoints[0].size(); j++) {
            cv::circle(img2, obj_origin_img[j], 8, { 255,0,0 }, -1);
            error = imgPoints2[idx2[i]][j] - obj_origin_img[j];
            std::cout << "error: " << error << std::endl;
            esum += sqrt(pow(error.x,2)+pow(error.y,2));
        }
        eave = esum / 30;
        esum = 0;
        std::cout << "eave2: " << eave << std::endl;
        cv::imwrite("./result4/" + substr, img2);
    
    }
    //for (int i = 0; i < idx.total(); i++) {
    for (int i = 0; i < 1; i++) {
        imgPath = imgList[idx2[i]];

        
        std::string substr = imgPath.substr(14);

        cv::omnidir::projectPoints(objPoints[0], obj_origin_img, rvecsL[i], tvecsL[i], K1, dxi[0], D1, yaco);

        img = cv::imread(imgPath);
        for (int j = 0; j < objPoints[0].size(); j++) {
            cv::circle(img, obj_origin_img[j], 8, { 255,0,0 }, -1);
            error = imgPoints[idx2[i]][j] - obj_origin_img[j];
            std::cout << "error: " << error << std::endl;
            esum += sqrt(pow(error.x, 2) + pow(error.y, 2));
        }
        eave = esum / 30;
        std::cout << "eave1: " << eave << std::endl;
        cv::imwrite("./Lresult/L" + substr, img);
    }
    
    

    
    
    

    
    std::cout << "Undistort" << std::endl;
    cv::Mat distorted = cv::imread("./image/Lcut4/cut00.png");
    cv::Mat undistorted, fishUndistorted, omniUndistorted;

    cv::omnidir::undistortImage(distorted, omniUndistorted, K1, D1, xi1, cv::omnidir::RECTIFY_PERSPECTIVE);

    cv::imwrite("./undistort_omnidir2.jpg", omniUndistorted);
    
    
}