
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
#include <iomanip>

namespace fs = std::experimental::filesystem;

std::vector<std::string> fileList(std::string dir, std::string ext = "")
{
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



    std::string imgDir = "./image/Lcut28";
    cv::Size patternSize(4, 3);
    cv::Size imgSize;

    int count = 0;
    std::vector<std::vector<cv::Point2f>> imgPoints;
    std::vector<std::vector<cv::Point3f>> objPoints;

    std::vector<std::string> imgList = fileList(imgDir, ".png");
    for (std::string imgPath : imgList) {
        
        std::cout << imgPath << "...";
        cv::Mat img = cv::imread(imgPath);
        std::string  str = std::to_string(count);
        std::vector<cv::Point2f> corners;
        bool found = cv::findChessboardCorners(img, patternSize, corners, cv::CALIB_CB_ADAPTIVE_THRESH + cv::CALIB_CB_NORMALIZE_IMAGE + cv::CALIB_CB_FAST_CHECK + cv::CALIB_CB_FILTER_QUADS);
        if (!found) {
            std::cout << "not found" << std::endl;
        }
        cv::drawChessboardCorners(img, cv::Size(4, 3), corners, true);
        cv::imwrite("./image/chess/chess12_"+str+".png",img);
        imgPoints.push_back(corners);
        std::cout << "found" << std::endl;
        imgSize = img.size();
        count++;
    }

    for (int i = 0;i < imgPoints.size();i++) {
        std::vector<cv::Point3f> obj;
        for (int c = 0; c < patternSize.height; c++) {
            for (int r = 0;r < patternSize.width; r++) {
                float x = r * 38.0;
                float y = c * 38.0;
                float z = 0.0;
                obj.push_back(cv::Point3f(x, y, z));
            }
        }
        objPoints.push_back(obj);

    }

    

    
    cv::Mat omniK, omniXi, omniD, omniR, omniT, idx;
    std::string omniFile = "./omniCalibrate28.xml";
    cv::TermCriteria critia(cv::TermCriteria::COUNT + cv::TermCriteria::EPS, 200, 0.0001);
    double omniRMS = cv::omnidir::calibrate(objPoints, imgPoints, imgSize, omniK, omniXi, omniD, omniR, omniT, 0, critia, idx);
    cv::FileStorage omnixml(omniFile, cv::FileStorage::WRITE);
    cv::write(omnixml, "RMS", omniRMS);
    cv::write(omnixml, "K", omniK);
    cv::write(omnixml, "Xi", omniXi);
    cv::write(omnixml, "D", omniD);
    cv::write(omnixml, "R", omniR);
    cv::write(omnixml, "T", omniT);
    cv::write(omnixml, "idx", idx);
    omnixml.release();
    std::cout << "K:" << omniK << std::endl;
    std::cout << "Xi:" << omniXi << std::endl;
    std::cout << "D:" << omniD << std::endl;
    std::cout << "R:" << omniR << std::endl;
    std::cout << "T:" << omniT << std::endl;
    std::cout << "idx:" << idx << std::endl;
    std::cout << "RMS:" << omniRMS << std::endl;
    



}