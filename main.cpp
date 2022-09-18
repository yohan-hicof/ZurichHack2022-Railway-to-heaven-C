#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include <chrono>
#include <iterator>
#include <numeric>
#include <cmath>

#include "dirent.h"

#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>

#include "skeletization.hpp"

using namespace std;
using namespace cv;

void process_single_image(cv::Mat &image, cv::Mat &mask, cv::Mat &output, float clearance_ratio = 1.0);
void draw_clearance(cv::Mat &mask, Point2f &pt1, Point2f &pt2, bool full = false);


struct rail_clearance {
    double inter_rail = 1835; //1435
    double inter_ratio = 1.0;
    vector<Point2f> list_points{
        Point2f(0,0),         Point2f(-1310,-90),   Point2f(-1580,-360),  Point2f(-1690,-360),
		Point2f(-1690,-560),  Point2f(-2100,-560),  Point2f(-2100,-1700), Point2f(-2200,-1700),
		Point2f(-2200,-3000), Point2f(-2100,-3000),	Point2f(-2100,-3330), Point2f(-1660,-4300),
		Point2f(-1020,-4800), Point2f(-1020,-6000),	Point2f(0,-6000),
		Point2f(1020,-6000),  Point2f(1020,-4800),
		Point2f(1660, -4300), Point2f(2100,-3330),  Point2f(2100,-3000),  Point2f(2200,-3000),
		Point2f(2200,-1700),  Point2f(2100,-1700),  Point2f(2100,-560),   Point2f(1690,-560),
		Point2f(1690,-360),   Point2f(1580,-360),   Point2f(1310,-90),    Point2f(0,0)};
};


vector<string> open_dir(string path) {

    DIR*    dir;
    dirent* pdir;
    vector<string> files;

    dir = opendir(path.c_str());

    while ((pdir = readdir(dir))) {
        files.push_back(pdir->d_name);
    }
    return files;
}

vector<string> convert_argc(int argc, char* argv[]){

    vector<string> retour;
    for (int i = 0; i < argc; i++){
        string curr = argv[i];
        retour.push_back(curr);
    }
    return retour;
}

void keep_rail_mask(cv::Mat &mask, cv::Mat &mask_rail){
    //Take the mask with 0 rail, 1 between rails, 2 background.
    //Create a clean mask where only rails connected to a between rail section is kept

    cv::Mat mask_copy = mask.clone();

    for (int y = 1; y < mask.rows; y++){
        for (int x = 1; x < mask.cols; x++){
            if (mask.at<unsigned char>(y,x) > 2) continue;
            if (mask.at<unsigned char>(y,x-1) == 0 && mask.at<unsigned char>(y,x) == 1){
                cv::floodFill(mask_copy, Point(x-1,y),Scalar(100));
                cv::floodFill(mask_copy, Point(x,y),Scalar(200));
                continue;
            }
            if (mask.at<unsigned char>(y,x-1) == 1 && mask.at<unsigned char>(y,x) == 0){
                cv::floodFill(mask_copy, Point(x-1,y),Scalar(200));
                cv::floodFill(mask_copy, Point(x,y),Scalar(100));
            }
        }
    }
    //Create an image with only the rails
    for (int y = 0; y < mask.rows; y++){
        for (int x = 0; x < mask.cols; x++){
            if (mask_copy.at<unsigned char>(y, x) == 100)
                mask_rail.at<unsigned char>(y, x) = 255;
        }
    }
}

void process_single_image(cv::Mat &image, cv::Mat &mask, cv::Mat &output, float clearance_ratio){

    /* The mask has only three values: 0 for rails, 1 for between rails, 2 for background
    */

    cv::Mat rail_only = cv::Mat::zeros(mask.size(), CV_8U);
    cv::Mat skeleton, clearance;

    auto start = chrono::high_resolution_clock::now();
    //Clean a bit the image.
    //Only keep the rails/background when we have both of them
    keep_rail_mask(mask, rail_only);
    //Copy the rail only and create the skeleton
    skeleton = rail_only.clone();
    thinning(skeleton);
    clearance = skeleton.clone();

    //Note, there will be an issue when I find several pairs of rails, shit
    /*Now we create the clearance mask
      For every single line in the image:
         -Find the rails (special case if only one), and the distance between them.
         -Create a mask between these two points + around depending on their distance
    */
    //First, we assume a single pair of rails
    for (int y = 0; y < skeleton.rows; y++){
        int pos1 = -1, pos2 = -1, distance;
        for (int x = 0; x < skeleton.cols; x++){
            if (skeleton.at<unsigned char>(y,x) != 0){
                if (pos1 == -1) pos1 = x;
                else if (x-pos1 > 2){//We need more than 1 pixel distance between the two points
                    pos2 = x;
                    break;
                }
            }
        }
        if (pos1 == -1) continue; //no rail found
        if (pos2 == -1) {cerr << "Single rail found, see later" << endl; continue;}
        distance = pos2-pos1;
        distance *= clearance_ratio;
        for (int x = max(0, pos1-distance); x < min(pos2+distance, clearance.cols); x++){
            clearance.at<unsigned char>(y, x) = 255;
        }
    }

    auto stop = chrono::high_resolution_clock::now();
    auto duration = chrono::duration_cast<chrono::milliseconds>(stop - start);
    cout << "Execution time: " << duration.count() << "ms" << endl;

    //Create an overlay image
    cv::cvtColor(clearance, output, COLOR_GRAY2BGR);
    cv::addWeighted(image, 0.5, output, 0.5, 0.0, output);


    cv::imshow("image", image);
    cv::imshow("original mask", mask);
    cv::imshow("Rail only", rail_only);
    cv::imshow("Skeleton", skeleton);
    cv::imshow("clearance", clearance);
    cv::imshow("output", output);
    cv::waitKey();

}

void process_single_image_V2(cv::Mat &image, cv::Mat &mask, cv::Mat &output){

    cv::Mat rail_only = cv::Mat::zeros(mask.size(), CV_8U);
    cv::Mat skeleton, clearance;
    //Clean a bit the image.
    //Only keep the rails/background when we have both of them
    keep_rail_mask(mask, rail_only);
    //Copy the rail only and create the skeleton
    skeleton = rail_only.clone();
    thinning(skeleton);

    for (int y = 700; y > 400; y--){
        Point2f pt1(-1,-1), pt2(-1,-1);
        for (int x = 0; x < skeleton.cols; x++){
            if (skeleton.at<unsigned char>(y, x) != 0){
                if (pt1.x == -1) pt1 = Point2f(x, y);
                else pt2 = Point2f(x, y);
            }
        }
        if (pt2.x > 0){
            draw_clearance(mask, pt1, pt2);
            Mat overlay, mask_rgb;
            cv::cvtColor(mask, mask_rgb, COLOR_GRAY2BGR);
            cv::addWeighted(image, 0.5, mask_rgb, 0.5, 0.0, overlay);
            imshow("mask", mask);
            imshow("image", image);
            imshow("overlay", overlay);
            waitKey(0);
        }
    }

}

void process_single_image_V3(cv::Mat &image, cv::Mat &mask, cv::Mat &output){

    cv::Mat blurry, gray, edges, inside_rail = cv::Mat::zeros(mask.size(), CV_8U);

    //Now the goal is to detect the stuff between the rails that might be // to the rail.
    //The goal is to be able to match the turning rails/with inclinaison

    for (int y = 0; y < mask.rows; y++){
        for (int x = 0; x < mask.cols; x++){
            if (mask.at<unsigned char>(y,x) == 1) inside_rail.at<unsigned char>(y, x) = 255;
        }
    }

    //Lets convert the image to hsv
    cv::Mat image_hsv, image_scaled = image.clone(), image_gray;

    double vmin, vmax;
    cv::cvtColor(image, image_gray, COLOR_BGR2GRAY);
    cv::minMaxIdx(image_gray, &vmin, &vmax, 0, 0, inside_rail);

    image_scaled -= vmin;
    image_scaled *= 255.0/(vmax-vmin);
    cerr << "Vmin: " << vmin << endl;
    cerr << "Vmax: " << vmax << endl;

    vector<cv::Mat> hsv;
    //cv::cvtColor(image, image_hsv, COLOR_BGR2YUV);
    cv::cvtColor(image_scaled, image_hsv, COLOR_BGR2HSV);
    cv::split(image_hsv, hsv);


    // Let try sobel edge detector
    cvtColor(image, gray, COLOR_BGR2GRAY);
    GaussianBlur(gray, blurry, Size(3,3), 0);
    Canny(blurry, edges, 100, 200, 3, false);
    edges &= inside_rail;

    //here the line detector should depends on the size between the rail we are currently considering.
    //Also, we will filter all the possible lines that are not roughtly perpendicular to the acutal rails.
    //Since we can assume that the rail are roughly vertical, all the line for horizontal.

    vector<Vec2f> lines; // will hold the results of the detection
    HoughLines(edges, lines, 1, CV_PI/180, 75, 0, 0 ); // runs the actual detection
    double angle_filter = 0.05;

    for( size_t i = 0; i < lines.size(); i++ ) {

        float rho = lines[i][0], theta = lines[i][1];
        //I check if tetha is close to pi/2 +- pi
        if ((theta > CV_PI/2-angle_filter && theta < CV_PI/2+angle_filter)||(-theta > CV_PI-angle_filter && -theta < CV_PI+angle_filter)){
            Point pt1, pt2;
            double a = cos(theta), b = sin(theta);
            double x0 = a*rho, y0 = b*rho;
            pt1.x = cvRound(x0 + 1000*(-b));
            pt1.y = cvRound(y0 + 1000*(a));
            pt2.x = cvRound(x0 - 1000*(-b));
            pt2.y = cvRound(y0 - 1000*(a));
            line( image, pt1, pt2, Scalar(0,0,255), 3, LINE_AA);
        }

    }
    mask *= 100;

    //cv::imshow("Mask", mask);
    //cv::imshow("Canny", edges);
    cv::imshow("Base", image);
    cv::imshow("Base hsv", image_hsv);
    cv::imshow("hsv[0]", hsv[0]);
    cv::imshow("hsv[1]", hsv[1]);
    cv::imshow("hsv[2]", hsv[2]);
    cv::waitKey();



}

void draw_clearance(cv::Mat &mask, Point2f &pt1, Point2f &pt2, bool full){

    // Draw the clearance in the mask using the two given points as the base point for the rail position.
    // We assume that the points are correct.
    // We scale the points of the clearance according to the distance between the two points.
    // Then we do the rotation according to the angle of the points to the ground.

    double distance, angle = 0, vec_size;
    Point2f centre, vec;
    rail_clearance rc;

    mask = cv::Mat::zeros(mask.size(), CV_8U);

    //Compute the center between the two points, the distance and their angle with the horizontal
    centre.x = (pt1.x+pt2.x)/2.0;
    centre.y = (pt1.y+pt2.y)/2.0;
    distance  = sqrt(pow(pt1.x-pt2.x, 2) + pow(pt1.y-pt2.y,2));

    vec.x = pt1.x-pt2.x;
    vec.y = pt1.y-pt2.y;
    vec_size  = sqrt(vec.x*vec.x + vec.y*vec.y);
    angle = acos(vec.x/vec_size);
    if (angle > CV_PI/2) angle -= CV_PI;
    else if (angle < -CV_PI/2) angle += CV_PI;

    rc.inter_ratio = distance/rc.inter_rail;

    // Resize, then rotate the points
    for (size_t i = 0; i < rc.list_points.size(); i++){
        Point2f curr = rc.list_points[i], rot;
        curr *= rc.inter_ratio;
        rot.x = (curr.x * cos(angle)) - (curr.y * sin(angle)) + centre.x;
        rot.y = (curr.x * sin(angle)) + (curr.y * cos(angle)) + centre.y;
        rc.list_points[i] = rot;
    }

    for (size_t i = 1; i < rc.list_points.size(); i++){
        cv::line(mask, rc.list_points[i-1], rc.list_points[i], Scalar(255,255,255), 1);
    }
    if (full){
        Point2f centre(mask.cols/2, mask.rows/4);
        centre = 0.5*(rc.list_points[0] + rc.list_points[14]);
        if (centre.x < 0 || centre.x >= mask.cols-1 || centre.y < 0 || centre.y >= mask.rows-1 || isnan(centre.x) || isnan(centre.y)) return;

        cv::floodFill(mask,centre, Scalar(255,255,255));
    }

}

void parse_argc(vector<string> &argv){

    //First, single image. Second folders of images
    string path_mask, path_image, path_out;
    string path_masks, path_images, path_outs;
    float clearance_ratio = 1.0;

    for (size_t i = 0; i < argv.size(); i++){
        if (argv[i] == "-m") {path_mask = argv[i+1]; i++; continue;}
        if (argv[i] == "-i") {path_image = argv[i+1]; i++; continue;}
        if (argv[i] == "-o") {path_out = argv[i+1]; i++; continue;}
        //Uppercase -> folders
        if (argv[i] == "-M") {path_masks = argv[i+1]; i++; continue;}
        if (argv[i] == "-I") {path_images = argv[i+1]; i++; continue;}
        if (argv[i] == "-O") {path_outs = argv[i+1]; i++; continue;}

        if (argv[i] == "-c") {clearance_ratio = stof(argv[i+1]); i++; continue;}
    }
    if (path_mask.size() > 0 && path_image.size() > 0 && path_out.size() > 0){
        Mat mask, image, output;
        mask = cv::imread(path_mask, 0);
        image = cv::imread(path_image, 1);
        if (mask.empty() || image.empty()){
            cerr << "Error, the mask or the image is not valid" << endl;
            return;
        }
        //process_single_image(image, mask, output, clearance_ratio);
        process_single_image_V2(image, mask, output);
        cerr << "Writting the output image at: " << path_out << endl;
        cv::imwrite(path_out, output);
    }

    if (path_masks.size() > 0 && path_images.size() > 0 && path_outs.size() > 0){
        //We assume that the same is actually the same in both path
        vector<string> list_masks = open_dir(path_masks);
        vector<string> list_images = open_dir(path_images);

        for (size_t i = 0; i < list_images.size(); i++){
            if (list_images[i].size() < 5) continue;
            Mat mask, image, output;
            mask = cv::imread(path_masks+list_masks[i], 0);
            image = cv::imread(path_images+list_masks[i], 1);
            if (mask.empty() || image.empty()){
                cerr << "Error, the mask or the image is not valid" << endl;
                cerr << "Mask name: " << path_masks+list_masks[i] << endl;
                cerr << "Image name: " << path_image+list_masks[i] << endl;
            }
            process_single_image(image, mask, output, clearance_ratio);
            cerr << "Writting the output image at: " << path_out << endl;
            cv::imwrite(path_outs+list_masks[i], output);
        }
    }
}


void overlay_single_image(cv::Mat &image, cv::Mat &mask, cv::Mat &output){

    cv::Mat rail_only = cv::Mat::zeros(mask.size(), CV_8U);
    cv::Mat skeleton, clearance;
    //Clean a bit the image.
    //Only keep the rails/background when we have both of them
    keep_rail_mask(mask, rail_only);
    //Copy the rail only and create the skeleton
    skeleton = rail_only.clone();
    thinning(skeleton);

    Point2f pt1(-1,-1), pt2(-1,-1);
    for (int x = 0; x < skeleton.cols; x++){
        if (skeleton.at<unsigned char>(600, x) != 0){
            if (pt1.x == -1) pt1 = Point2f(x, 600);
            else pt2 = Point2f(x, 600);
        }
    }
    if (pt2.x > 0){
        draw_clearance(mask, pt1, pt2, true);
        Mat mask_rgb;
        cv::cvtColor(mask, mask_rgb, COLOR_GRAY2BGR);
        cv::addWeighted(image, 0.5, mask_rgb, 0.5, 0.0, output);
    }

}

void get_all_pair_points(vector<string> &list_mask, string &path_mask, vector<Point2f> &list_pt, vector<string> &list_name){

    list_pt.clear();


    for (size_t i = 0; i < list_mask.size(); i++){
        cv::Mat mask = imread(path_mask+list_mask[i],0);
        if (mask.empty())continue;
        cerr << list_mask[i] << endl;
        cv::Mat rail_only = cv::Mat::zeros(mask.size(), CV_8U);
        cv::Mat skeleton, clearance;
        //Clean a bit the image.
        //Only keep the rails/background when we have both of them
        keep_rail_mask(mask, rail_only);
        //Copy the rail only and create the skeleton
        skeleton = rail_only.clone();
        thinning(skeleton);

        Point2f pt1(-1,-1), pt2(-1,-1);
        for (int x = 0; x < skeleton.cols; x++){
            if (skeleton.at<unsigned char>(600, x) != 0){
                if (pt1.x == -1) pt1 = Point2f(x, 600);
                else pt2 = Point2f(x, 600);
            }
        }
        if (pt2.x > 0){
            list_pt.push_back(pt1);
            list_pt.push_back(pt2);
            list_name.push_back(list_mask[i]);
        }
        else{
            list_pt.push_back(Point2f(-1,-1));
            list_pt.push_back(Point2f(-1,-1));
            list_name.push_back(list_mask[i]);
        }

    }

    ofstream fichier;
    fichier.open("./list_pt.csv");
    for (size_t i = 0; i < list_name.size(); i++){
        fichier << list_name[i] << " " << list_pt[2*i].x << " " << list_pt[2*i].y << " "<< list_pt[2*i+1].x << " " << list_pt[2*i+1].y << "\n";

    }
    fichier.close();
}

void clean_pair_points(vector<Point2f> &list_pt){

    //Compute the distance for all the pair of points (i,i+1)
    //When there is a jump between consecutive distance, check how long it last.
    //If short use the previous/next to average.

    vector<double> list_dist;
    double sum_dist=0, cpt_dist=0, mean_dist = 0;

    for (size_t i = 0; i < list_pt.size(); i+=2){
        double dist = sqrt(pow(list_pt[i].x-list_pt[i+1].x,2)+pow(list_pt[i].y-list_pt[i+1].y,2));
        list_dist.push_back(dist);
        if (dist > 5){sum_dist+=dist; cpt_dist++;}
    }

    mean_dist = sum_dist/cpt_dist;

    for (size_t i = 1; i < list_dist.size()-1; i++){
        if (list_dist[i] < 5) continue;
        double rat = min(list_dist[i-1],list_dist[i])/max(list_dist[i-1],list_dist[i]);
        if (rat < 0.8){
            if (fabs(list_dist[i]-mean_dist) < fabs(list_dist[i-1]-mean_dist)){
                list_pt[2*i-2] = list_pt[2*i];
                list_pt[2*i-1] = list_pt[2*i+1];
                list_dist[i-1] = list_dist[i];
            }
            else{
                list_pt[2*i] = list_pt[2*i-2];
                list_pt[2*i+1] = list_pt[2*i-1];
                list_dist[i] = list_dist[i-1];
            }
        }

    }


}

void create_video(){

    string path_frames = "/home/yohan/Documents/data/download/frames/";
    string path_masks = "/home/yohan/Documents/data/download/masks/";
    string path_output = "/home/yohan/Documents/data/download/outputs/";

    vector<string> list_images = open_dir(path_frames);

    sort(list_images.begin(), list_images.end());

    for (size_t i = 0; i < list_images.size(); i++){
        if (list_images[i].size() < 5) continue;
        cerr << list_images[i] << endl;

        cv::Mat frame = imread(path_frames+list_images[i],1);
        cv::Mat mask = imread(path_masks+list_images[i],0);
        cv::Mat output = cv::Mat::zeros(mask.size(), CV_8UC3);

        if (frame.empty() || mask.empty()) continue;

        overlay_single_image(frame, mask, output);
        cv::imwrite(path_output+list_images[i], output);


    }

}

void create_video_v2(){

    string path_frames = "/home/yohan/Documents/data/download/old/frames/";
    string path_masks = "/home/yohan/Documents/data/download/old/masks/";
    string path_output = "/home/yohan/Documents/data/download/old/outputs/";
    vector<Point2f> list_pt;

    vector<string> list_images = open_dir(path_frames);
    vector<string> list_name;
    sort(list_images.begin(), list_images.end());

    if (false)
        get_all_pair_points(list_images, path_masks, list_pt, list_name);
    else{
        ifstream fichier;
        string name;
        int x1, y1, x2, y2;
        fichier.open("./list_pt_.csv");
        while (fichier >> name >> x1 >> y1 >> x2 >> y2){
            list_name.push_back(name);
            list_pt.push_back(Point2f(x1,y1));
            list_pt.push_back(Point2f(x2,y2));
        }
        fichier.close();
    }

    clean_pair_points(list_pt);
    cerr << "Creating the overlay" << endl;
    //smooth the list of points
    for (size_t i = 2; i < list_name.size()-2; i++){
        cerr << list_name[i] << endl;
        Point2f pt1 = (list_pt[2*(i-2)]+list_pt[2*(i-1)]+list_pt[2*i]+list_pt[2*(i+2)]+list_pt[2*(i+1)])*0.2;
        Point2f pt2 = (list_pt[2*(i-2)+1]+list_pt[2*(i-1)+1]+list_pt[2*i+1]+list_pt[2*(i+2)+1]+list_pt[2*(i+1)+1])*0.2;

        cv::Mat frame = imread(path_frames+list_name[i],1);
        cv::Mat mask = cv::Mat::zeros(frame.size(), CV_8U);
        cv::Mat output = cv::Mat::zeros(mask.size(), CV_8UC3);

        if (mask.empty() || frame.empty()) continue;

        draw_clearance(mask, pt1, pt2, true);
        cv::cvtColor(mask, mask, COLOR_GRAY2BGR);
        cv::addWeighted(mask, 0.5, frame, 0.5, 0.0, output);
        cv::imwrite(path_output+list_name[i], output);

    }


}


void create_video_v3(){

    string path_frames = "/home/yohan/Documents/data/download/frames/";
    string path_masks = "/home/yohan/Documents/data/download/masks/";
    //string path_map = "/home/yohan/Documents/data/download/unsquized/";
    string path_map = "/home/yohan/Documents/data/download/dist_map/";
    string path_output = "/home/yohan/Documents/data/download/outputs/";
    vector<Point2f> list_pt;

    vector<string> list_images = open_dir(path_frames);
    vector<string> list_name;
    sort(list_images.begin(), list_images.end());

    if (false)
        get_all_pair_points(list_images, path_masks, list_pt, list_name);
    else{
        ifstream fichier;
        string name;
        int x1, y1, x2, y2;
        fichier.open("./list_pt_old.csv");
        while (fichier >> name >> x1 >> y1 >> x2 >> y2){
            list_name.push_back(name);
            list_pt.push_back(Point2f(x1,y1));
            list_pt.push_back(Point2f(x2,y2));
        }
        fichier.close();
    }

    clean_pair_points(list_pt);
    cerr << "Creating the overlay" << endl;
    //smooth the list of points
    for (size_t i = 2; i < list_name.size()-2; i++){
        cerr << list_name[i] << endl;
        Point2f pt1 = (list_pt[2*(i-2)]+list_pt[2*(i-1)]+list_pt[2*i]+list_pt[2*(i+2)]+list_pt[2*(i+1)])*0.2;
        Point2f pt2 = (list_pt[2*(i-2)+1]+list_pt[2*(i-1)+1]+list_pt[2*i+1]+list_pt[2*(i+2)+1]+list_pt[2*(i+1)+1])*0.2;

        cv::Mat frame = imread(path_frames+list_name[i],1);
        cv::Mat mask = cv::Mat::zeros(frame.size(), CV_8U);
        cv::Mat output = cv::Mat::zeros(mask.size(), CV_8UC3);

        //string heat_name = path_map + list_name[i].substr(0, list_name[i].size()-4) + "_disp.jpeg";
        string heat_name = path_map + list_name[i].substr(0, list_name[i].size()-4) + "_disp.png";
        cv::Mat heatmap = imread(heat_name, 1);

        if (mask.empty() || frame.empty() || heatmap.empty()) continue;
        draw_clearance(mask, pt1, pt2, true);
        cv::cvtColor(mask, mask, COLOR_GRAY2BGR);
        //truncate the dist map to keep value between 130 and 150
        for (int y = 0; y < heatmap.rows; y++){
            for (int x = 0; x < heatmap.cols; x++){
                if (heatmap.at<Vec3b>(y,x)[2] > 150) heatmap.at<Vec3b>(y,x)[2] = 0;
                if (heatmap.at<Vec3b>(y,x)[2] < 120) heatmap.at<Vec3b>(y,x)[2] = 0;
                heatmap.at<Vec3b>(y,x)[0] = 0;
                heatmap.at<Vec3b>(y,x)[1] = 0;
            }
        }

        heatmap &= mask;
        cv::addWeighted(mask, 0.5, frame, 0.5, 0.0, frame);
        cv::addWeighted(heatmap, 0.5, frame, 0.5, 0.0, output);
        cv::imwrite(path_output+list_name[i], output);
    }


}

void create_single_image_video(){

    string path_frames = "/home/yohan/Documents/data/siemens/pictures/20220712_073943.jpg";
    string path_masks = "/home/yohan/Documents/data/siemens/pictures/20220712_073943_mask.png";
    string path_output = "/home/yohan/Documents/data/siemens/pictures/outputs/";

    cv::Mat frame = cv::imread(path_frames,1);
    cv::Mat mask = cv::imread(path_masks,0);

    cv::Mat rail_only = cv::Mat::zeros(mask.size(), CV_8U);
    cv::Mat skeleton, output;
    //Clean a bit the image.
    //Only keep the rails/background when we have both of them
    keep_rail_mask(mask, rail_only);
    //Copy the rail only and create the skeleton
    skeleton = rail_only.clone();
    thinning(skeleton);

    cerr << "Skeletization done" << endl;

    int cpt = 1;
    for (int y = 2450; y > 1450; y--){
        if (y%50 == 49)
            cerr << "position: " << y << endl;

        Point2f pt1(-1,-1), pt2(-1,-1);
        for (int x = 0; x < skeleton.cols; x++){
            if (skeleton.at<unsigned char>(y, x) != 0){
                if (pt1.x == -1) pt1 = Point2f(x, y);
                else pt2 = Point2f(x, y);
            }
        }
        if (pt2.x > 0){
            draw_clearance(mask, pt1, pt2, true);
            Mat mask_rgb;
            cv::cvtColor(mask, mask_rgb, COLOR_GRAY2BGR);
            cv::addWeighted(frame, 0.5, mask_rgb, 0.5, 0.0, output);
            string out_name = path_output + "overlay_";
            if (cpt < 10) out_name += "0";
            if (cpt < 100) out_name += "0";
            if (cpt < 1000) out_name += "0";
            out_name += to_string(cpt++) + ".png";
            cv::imwrite(out_name, output);
        }
    }
}

void resize_all(){

    string path_output = "/home/yohan/Documents/data/siemens/pictures/outputs/";
    string path_resized = "/home/yohan/Documents/data/siemens/pictures/resized/";

    vector<string> list_images = open_dir(path_output);
    sort(list_images.begin(), list_images.end());

    for (size_t i = 0; i < list_images.size(); i++){
        if (i%50 == 49) cerr << i << endl;
        cv::Mat image = imread(path_output+list_images[i],1);
        if (image.empty()) continue;
        cv::resize(image, image, Size(1280,960), 0, 0, INTER_NEAREST);
        cv::imwrite(path_resized+list_images[i], image);
    }

}

void resize_crop_all(){

    string path_in = "/home/yohan/Documents/data/download/frames/";
    string path_resized = "/home/yohan/Documents/data/download/cropped/";

    vector<string> list_images = open_dir(path_in);
    sort(list_images.begin(), list_images.end());

    for (size_t i = 0; i < list_images.size(); i++){
        if (i%50 == 49) cerr << i << endl;
        cv::Mat image = imread(path_in+list_images[i],1);
        if (image.empty()) continue;
        //double fx = 1024.0/image.cols;
        //cv::resize(image, image, Size(0,0), fx, fx, INTER_NEAREST);
        //image = image(Rect(Point(0,image.rows/2-160),Point(1024,image.rows/2+160))).clone();
        cv::resize(image, image, Size(1024,320));
        cv::imwrite(path_resized+list_images[i], image);
    }

}

void unsquish_all(){

    string path_out = "/home/yohan/Documents/data/download/unsquized/";
    string path_in = "/home/yohan/Documents/data/download/cropped/";

    vector<string> list_images = open_dir(path_in);
    sort(list_images.begin(), list_images.end());

    for (size_t i = 0; i < list_images.size(); i++){
        if (i%50 == 49) cerr << i << endl;
        if (list_images[i].find("disp.jpeg")==string::npos) continue;
        cv::Mat image = imread(path_in+list_images[i],1);
        if (image.empty()) continue;
        cv::resize(image, image, Size(1280,720));
        cv::imwrite(path_out+list_images[i], image);
    }

}

string gen_name(size_t i){

    string path_cropped = "/home/yohan/Documents/data/download/cropped/";
    string nb_zero = "0";
    if (i < 10) nb_zero += "0";
    if (i < 100) nb_zero += "0";
    if (i < 1000) nb_zero += "0";

    return path_cropped + "images" + nb_zero + to_string(i) + "_disp.jpeg";

}

void create_dist_overlay(){

    //string path_frame = "/home/yohan/Documents/data/download/cropped/";
    string path_frame = "/home/yohan/Documents/data/download/frames/";
    string path_out = "/home/yohan/Documents/data/download/distance/";

    vector<string> list_images = open_dir(path_frame);
    sort(list_images.begin(), list_images.end());

    for (size_t i = 12; i < 1750; i++){
        string nb_zero = "0";
        if (i < 10) nb_zero += "0";
        if (i < 100) nb_zero += "0";
        if (i < 1000) nb_zero += "0";
        string path_frame = path_frame + "images" + nb_zero + to_string(i) + ".png";
        string path_result = path_out + "images" + nb_zero + to_string(i) + ".png";
        string path_over1 = gen_name(i-1);
        string path_over2 = gen_name(i);
        string path_over3 = gen_name(i+1);

        cv::Mat frame = cv::imread(path_frame,1);
        cv::Mat over1 = cv::imread(path_over1,1);
        cv::Mat over2 = cv::imread(path_over2,1);
        cv::Mat over3 = cv::imread(path_over3,1);

        cv::addWeighted(over1, 0.5, over3, 0.5, 0.0, over1);
        cv::addWeighted(over1, 0.5, over2, 0.5, 0.0, over2);
        cv::addWeighted(frame, 0.5, over2, 0.5, 0.0, frame);

        cv::imwrite(path_result, frame);

    }

}


int main(int argc, char* argv[]){

    //create_video(); return 0;
    //create_video_v2(); return 0;
    create_video_v3();
    //create_single_image_video(); return 0;
    //resize_all(); return 0;
    //resize_crop_all();return 0;
    //create_dist_overlay(); return 0;
    //unsquish_all();

    //test_single_image();
    return 0;
}
