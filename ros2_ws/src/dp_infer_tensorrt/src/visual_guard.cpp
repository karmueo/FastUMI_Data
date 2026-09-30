// 腕部图像的球心定位；阈值和连通域选择顺序与 dp_infer.visual_guard 一致。
#include "dp_infer_tensorrt/core.hpp"

#include <algorithm>
#include <cmath>
#include <stdexcept>

#include <opencv2/imgproc.hpp>

namespace dp_infer_tensorrt {
namespace {
struct Marker { int area; cv::Point2d center; };
std::optional<cv::Point2d> marked_center(const cv::Mat &white, const cv::Mat &blue,
                                       const cv::Mat &orange, double scale) {
    std::array<std::vector<Marker>,2> markers;
    const std::array<cv::Mat,2> masks{blue,orange};
    const int min_area[2] = {40,80};
    for (int m = 0; m < 2; ++m) {
        cv::Mat labels, stats, centers;
        int count = cv::connectedComponentsWithStats(masks[m], labels, stats, centers);
        for (int i = 1; i < count; ++i) {
            int width = stats.at<int>(i,cv::CC_STAT_WIDTH), height = stats.at<int>(i,cv::CC_STAT_HEIGHT);
            int area = stats.at<int>(i,cv::CC_STAT_AREA);
            cv::Point2d center(centers.at<double>(i,0),centers.at<double>(i,1));
            if (area >= std::max(2.0,min_area[m]*scale*scale) && width >= 10*scale && width <= 140*scale &&
                height >= 10*scale && height <= 140*scale && center.x > white.cols*.1 &&
                center.x < white.cols*.9 && center.y < white.rows*.78)
                markers[m].push_back({area,center});
        }
    }
    int radius = std::max(3,static_cast<int>(std::nearbyint(90*scale))), best = -1;
    std::optional<cv::Point2d> result;
    for (const auto &b : markers[0]) for (const auto &o : markers[1]) {
        if (cv::norm(b.center-o.center) > 110*scale) continue;
        auto center = (b.center+o.center)*.5;
        int x = static_cast<int>(center.x), y = static_cast<int>(center.y);
        int left = std::max(0,x-radius), top = std::max(0,y-radius);
        cv::Rect rect(left,top,std::min(white.cols,x+radius)-left,std::min(white.rows,y+radius)-top);
        int score = std::min(b.area,o.area);
        if (cv::countNonZero(white(rect)) >= std::max(20.0,1000*scale*scale) && score > best) {
            best = score; result = center;
        }
    }
    return result;
}
}  // namespace

std::optional<cv::Point2d> ball_center_bgr(const cv::Mat &bgr) {
    if (bgr.empty() || bgr.type() != CV_8UC3) throw std::invalid_argument("Expected HWC uint8 BGR");
    const int width = bgr.cols, height = bgr.rows;
    if (width > 640) {
        int resized_height = std::max(1,static_cast<int>(std::nearbyint(height*640.0/width)));
        cv::Mat resized;
        cv::resize(bgr,resized,cv::Size(640,resized_height),0,0,cv::INTER_AREA);
        auto center = ball_center_bgr(resized);
        if (center) return cv::Point2d(center->x*width/640.0,center->y*height/resized_height);
        return std::nullopt;
    }
    const double scale = width/1280.0;
    cv::Mat hsv,white,orange,blue;
    cv::cvtColor(bgr,hsv,cv::COLOR_BGR2HSV);
    std::vector<int> values;
    for (int y = 0; y < height; y += 16) for (int x = 0; x < width; x += 16)
        values.push_back(hsv.at<cv::Vec3b>(y,x)[2]);
    std::sort(values.begin(),values.end());
    double index = (values.size()-1)*.95;
    size_t lower = static_cast<size_t>(index), upper = static_cast<size_t>(std::ceil(index));
    double percentile = values[lower] + (index-lower)*(values[upper]-values[lower]);
    cv::inRange(hsv,cv::Scalar(0,0,percentile < 100 ? 120 : 155),cv::Scalar(179,40,255),white);
    white.rowRange(static_cast<int>(height*.82),height).setTo(0);
    cv::inRange(hsv,cv::Scalar(4,90,60),cv::Scalar(30,255,255),orange);
    cv::inRange(hsv,cv::Scalar(90,80,25),cv::Scalar(135,255,255),blue);
    cv::Mat labels,stats,centers;
    int count = cv::connectedComponentsWithStats(white,labels,stats,centers);
    std::vector<int> choices,near_jaw;
    for (int i = 1; i < count; ++i) {
        int x = stats.at<int>(i,cv::CC_STAT_LEFT), y = stats.at<int>(i,cv::CC_STAT_TOP);
        int w = stats.at<int>(i,cv::CC_STAT_WIDTH), h = stats.at<int>(i,cv::CC_STAT_HEIGHT);
        int area = stats.at<int>(i,cv::CC_STAT_AREA);
        cv::Point2d center(centers.at<double>(i,0),centers.at<double>(i,1));
        if (area < std::max(8.0,1000*scale*scale) || w < 30*scale || w > 180*scale ||
            h < 30*scale || h > 180*scale || center.x <= width*.1 || center.x >= width*.9 || center.y >= height*.8)
            continue;
        int padding = std::max(2,static_cast<int>(std::nearbyint(15*scale)));
        int left = std::max(0,x-padding), top = std::max(0,y-padding);
        cv::Rect rect(left,top,std::min(width,x+w+padding)-left,std::min(height,y+h+padding)-top);
        if (cv::countNonZero(blue(rect)) >= std::max(2.0,40*scale*scale) &&
            cv::countNonZero(orange(rect)) >= std::max(4.0,100*scale*scale)) {
            choices.push_back(i);
            if (cv::norm(center-cv::Point2d(width*.5,height*.67)) < width*.2) near_jaw.push_back(i);
        }
    }
    if (choices.empty()) return marked_center(white,blue,orange,scale);
    const auto &candidates = near_jaw.empty() ? choices : near_jaw;
    int selected = *std::max_element(candidates.begin(),candidates.end(),[&stats](int a,int b) {
        return stats.at<int>(a,cv::CC_STAT_AREA) < stats.at<int>(b,cv::CC_STAT_AREA);
    });
    return cv::Point2d(centers.at<double>(selected,0),centers.at<double>(selected,1));
}
bool ball_near_jaws(const std::optional<cv::Point2d> &center,int width,int height) {
    if (!center || width <= 0 || height <= 0 || !std::isfinite(center->x) || !std::isfinite(center->y)) return false;
    return cv::norm(*center-cv::Point2d(width*.5,height*.67)) <= width*.0625;
}
}  // namespace dp_infer_tensorrt
