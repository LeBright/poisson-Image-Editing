#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/photo.hpp>
#include <opencv2/opencv.hpp>
#include <iostream>
#include <vector>
#include <map>
#include <iterator>

int main()
{
    // Init Mat
    cv::Mat Origin = cv::imread("test/src_img01.jpg");
    cv::Mat Target = cv::imread("test/tar_img01.jpg");
    cv::Mat Mask = cv::imread("test/mask_img01.jpg",0);
    cv::Mat Mask_Boundary = cv::Mat::zeros(Mask.size(), CV_32FC1);
    cv::Mat Origin_GradX(Origin.size(), CV_32FC3, cv::Scalar(0, 0, 0));
    cv::Mat Origin_GradY(Origin.size(), CV_32FC3, cv::Scalar(0, 0, 0));
    cv::Mat Target_GradX(Target.size(), CV_32FC3, cv::Scalar(0, 0, 0));
    cv::Mat Target_GradY(Target.size(), CV_32FC3, cv::Scalar(0, 0, 0));
    cv::Mat GradX(Target.size(), CV_32FC3, cv::Scalar(0, 0, 0));
    cv::Mat GradY(Target.size(), CV_32FC3, cv::Scalar(0, 0, 0));
    cv::Mat LapX(Target.size(), CV_32FC3, cv::Scalar(0, 0, 0));
    cv::Mat LapY(Target.size(), CV_32FC3, cv::Scalar(0, 0, 0));
    cv::Mat Lap(Target.size(), CV_32FC3, cv::Scalar(0, 0, 0));
    cv::Point Location(10, 10);
    int Dim = 0;

    // Check
    int error = 0;
    try {
        if (Mask.rows != Origin.rows || Mask.cols != Origin.cols)
            error = 1;
        else if (Location.x + Origin.rows > Target.rows || Location.y + Origin.cols > Target.cols)
            error = 2;
        throw error;
    }
    catch (int)
    {
        if (error == 1)
        {
            std::cout << "Mask and Origin has different size!" << std::endl;
            return error;
        }
        else if (error == 2)
        {
            std::cout << "Editing scope is out of target scope!" << std::endl;
            return error;
        }
    }
    
    // Find Mask's boundary
    for (int i = 0; i < Mask.rows; i++)
    {
        for (int j = 0; j < Mask.cols; j++)
        {
            Mask.at<uchar>(i, j) = Mask.at<uchar>(i, j) > 200 ? 255 : 0;
            if (Mask.at<uchar>(i, j) == 255)
            {
                Dim++;
            }
        }
    }
    for (int i = 0; i < Mask.rows; i++)
    {
        for (int j = 0; j < Mask.cols; j++)
        {
            uchar up = i == 0 ? 0 : Mask.at<uchar>(i - 1, j);
            uchar down = i == Mask.rows - 1 ? 0 : Mask.at<uchar>(i + 1, j);
            uchar left = j == 0 ? 0 : Mask.at<uchar>(i, j - 1);
            uchar right = j == Mask.cols - 1 ? 0 : Mask.at<uchar>(i, j + 1);
            uchar self = Mask.at<uchar>(i, j);
            Mask_Boundary.at<float>(i, j) = (self == up && self == down && self == left && self == right) ? 0 : 255;
        }
    }

    // Calculate b
    std::map<int,std::pair<int, int>>Index2Point;
    std::map<std::pair<int, int>, int>Point2Index;
    std::vector<std::pair<int, int>>Index_in_Target;
    cv::Mat b(Dim, 1, CV_32FC3, cv::Scalar(0, 0, 0));
    int index = 0;
    for (int i = 0; i < Mask.rows; i++)
    {
        for (int j = 0; j < Mask.cols; j++)
        {
            if (Mask.at<uchar>(i, j))
            {
                Origin_GradX.at<cv::Vec3f>(i, j)[0] = 
                    i == Mask.rows - 1 ? 0 - (float)Origin.at<cv::Vec3b>(i, j)[0] : (float)Origin.at<cv::Vec3b>(i + 1, j)[0] - (float)Origin.at<cv::Vec3b>(i, j)[0];
                Origin_GradX.at<cv::Vec3f>(i, j)[1] =
                    i == Mask.rows - 1 ? 0 - (float)Origin.at<cv::Vec3b>(i, j)[1] : (float)Origin.at<cv::Vec3b>(i + 1, j)[1] - (float)Origin.at<cv::Vec3b>(i, j)[1];
                Origin_GradX.at<cv::Vec3f>(i, j)[2] =
                    i == Mask.rows - 1 ? 0 - (float)Origin.at<cv::Vec3b>(i, j)[2] : (float)Origin.at<cv::Vec3b>(i + 1, j)[2] - (float)Origin.at<cv::Vec3b>(i, j)[2];
                Origin_GradY.at<cv::Vec3f>(i, j)[0] =
                    j == Mask.cols - 1 ? 0 - (float)Origin.at<cv::Vec3b>(i, j)[0] : (float)Origin.at<cv::Vec3b>(i, j + 1)[0] - (float)Origin.at<cv::Vec3b>(i, j)[0];
                Origin_GradY.at<cv::Vec3f>(i, j)[1] =
                    j == Mask.cols - 1 ? 0 - (float)Origin.at<cv::Vec3b>(i, j)[1] : (float)Origin.at<cv::Vec3b>(i, j + 1)[1] - (float)Origin.at<cv::Vec3b>(i, j)[1];
                Origin_GradY.at<cv::Vec3f>(i, j)[2] =
                    j == Mask.cols - 1 ? 0 - (float)Origin.at<cv::Vec3b>(i, j)[2] : (float)Origin.at<cv::Vec3b>(i, j + 1)[2] - (float)Origin.at<cv::Vec3b>(i, j)[2];

                int i_target, j_target;
                i_target = i + Location.x;
                j_target = j + Location.y;
                Target_GradX.at<cv::Vec3f>(i_target, j_target)[0] =
                    i_target == Target.rows - 1 ?
                    0 - (float)Target.at<cv::Vec3b>(i_target, j_target)[0] : (float)Target.at<cv::Vec3b>(i_target + 1, j_target)[0] - (float)Target.at<cv::Vec3b>(i_target, j_target)[0];
                Target_GradX.at<cv::Vec3f>(i_target, j_target)[1] =
                    i_target == Target.rows - 1 ? 
                    0 - (float)Target.at<cv::Vec3b>(i_target, j_target)[1] : (float)Target.at<cv::Vec3b>(i_target + 1, j_target)[1] - (float)Target.at<cv::Vec3b>(i_target, j_target)[1];
                Target_GradX.at<cv::Vec3f>(i_target, j_target)[2] =
                    i_target == Target.rows - 1 ? 
                    0 - (float)Target.at<cv::Vec3b>(i_target, j_target)[2] : (float)Target.at<cv::Vec3b>(i_target + 1, j_target)[2] - (float)Target.at<cv::Vec3b>(i_target, j_target)[2];
                
                Target_GradY.at<cv::Vec3f>(i_target, j_target)[0] =
                    j_target == Target.cols - 1 ?
                    0 - (float)Target.at<cv::Vec3b>(i_target, j_target)[0] : (float)Target.at<cv::Vec3b>(i_target, j_target + 1)[0] - (float)Target.at<cv::Vec3b>(i_target, j_target)[0];
                Target_GradY.at<cv::Vec3f>(i_target, j_target)[1] =
                    j_target == Target.cols - 1 ?
                    0 - (float)Target.at<cv::Vec3b>(i_target, j_target)[1] : (float)Target.at<cv::Vec3b>(i_target, j_target + 1)[1] - (float)Target.at<cv::Vec3b>(i_target, j_target)[1];
                Target_GradY.at<cv::Vec3f>(i_target, j_target)[2] =
                    j_target == Target.cols - 1 ?
                    0 - (float)Target.at<cv::Vec3b>(i_target, j_target)[2] : (float)Target.at<cv::Vec3b>(i_target, j_target + 1)[2] - (float)Target.at<cv::Vec3b>(i_target, j_target)[2];
                
                GradX.at<cv::Vec3f>(i_target, j_target)[0] = Origin_GradX.at<cv::Vec3f>(i, j)[0] + Target_GradX.at<cv::Vec3f>(i_target, j_target)[0];
                GradX.at<cv::Vec3f>(i_target, j_target)[1] = Origin_GradX.at<cv::Vec3f>(i, j)[1] + Target_GradX.at<cv::Vec3f>(i_target, j_target)[1];
                GradX.at<cv::Vec3f>(i_target, j_target)[2] = Origin_GradX.at<cv::Vec3f>(i, j)[2] + Target_GradX.at<cv::Vec3f>(i_target, j_target)[2];
                GradY.at<cv::Vec3f>(i_target, j_target)[0] = Origin_GradY.at<cv::Vec3f>(i, j)[0] + Target_GradY.at<cv::Vec3f>(i_target, j_target)[0];
                GradY.at<cv::Vec3f>(i_target, j_target)[1] = Origin_GradY.at<cv::Vec3f>(i, j)[1] + Target_GradY.at<cv::Vec3f>(i_target, j_target)[1];
                GradY.at<cv::Vec3f>(i_target, j_target)[2] = Origin_GradY.at<cv::Vec3f>(i, j)[2] + Target_GradY.at<cv::Vec3f>(i_target, j_target)[2];
            }
        }
    }
    for (int i = 0; i < Mask.rows; i++)
    {
        for (int j = 0; j < Mask.cols; j++)
        {
            if (Mask.at<uchar>(i, j))
            {
                int i_target, j_target;
                i_target = i + Location.x;
                j_target = j + Location.y;
                LapX.at<cv::Vec3f>(i_target, j_target)[0] =
                    i_target == Mask.rows - 1 ?
                    0 - GradX.at<cv::Vec3f>(i_target, j_target)[0] : GradX.at<cv::Vec3f>(i_target + 1, j_target)[0] - GradX.at<cv::Vec3f>(i_target, j_target)[0];
                LapX.at<cv::Vec3f>(i_target, j_target)[1] =
                    i_target == Mask.rows - 1 ?
                    0 - GradX.at<cv::Vec3f>(i_target, j_target)[1] : GradX.at<cv::Vec3f>(i_target + 1, j_target)[1] - GradX.at<cv::Vec3f>(i_target, j_target)[1];
                LapX.at<cv::Vec3f>(i_target, j_target)[2] =
                    i_target == Mask.rows - 1 ?
                    0 - GradX.at<cv::Vec3f>(i_target, j_target)[2] : GradX.at<cv::Vec3f>(i_target + 1, j_target)[2] - GradX.at<cv::Vec3f>(i_target, j_target)[2];
                LapY.at<cv::Vec3f>(i_target, j_target)[0] =
                    j_target == Mask.cols - 1 ?
                    0 - GradY.at<cv::Vec3f>(i_target, j_target)[0] : GradY.at<cv::Vec3f>(i_target, j_target + 1)[0] - GradY.at<cv::Vec3f>(i_target, j_target)[0];
                LapY.at<cv::Vec3f>(i_target, j_target)[1] =
                    j_target == Mask.cols - 1 ?
                    0 - GradY.at<cv::Vec3f>(i_target, j_target)[1] : GradY.at<cv::Vec3f>(i_target, j_target + 1)[1] - GradY.at<cv::Vec3f>(i_target, j_target)[1];
                LapY.at<cv::Vec3f>(i_target, j_target)[2] =
                    j_target == Mask.cols - 1 ?
                    0 - GradY.at<cv::Vec3f>(i_target, j_target)[2] : GradY.at<cv::Vec3f>(i_target, j_target + 1)[2] - GradY.at<cv::Vec3f>(i_target, j_target)[2];

                Lap.at<cv::Vec3f>(i_target, j_target)[0] = LapX.at<cv::Vec3f>(i_target, j_target)[0] + LapY.at<cv::Vec3f>(i_target, j_target)[0];
                Lap.at<cv::Vec3f>(i_target, j_target)[1] = LapX.at<cv::Vec3f>(i_target, j_target)[1] + LapY.at<cv::Vec3f>(i_target, j_target)[1];
                Lap.at<cv::Vec3f>(i_target, j_target)[2] = LapX.at<cv::Vec3f>(i_target, j_target)[2] + LapY.at<cv::Vec3f>(i_target, j_target)[2];

                b.at<cv::Vec3f>(index, 0)[0] = Mask_Boundary.at<float>(i, j) ? (float)Target.at<cv::Vec3b>(i_target, j_target)[0] : Lap.at<cv::Vec3f>(i_target, j_target)[0];
                b.at<cv::Vec3f>(index, 0)[1] = Mask_Boundary.at<float>(i, j) ? (float)Target.at<cv::Vec3b>(i_target, j_target)[1] : Lap.at<cv::Vec3f>(i_target, j_target)[1];
                b.at<cv::Vec3f>(index, 0)[2] = Mask_Boundary.at<float>(i, j) ? (float)Target.at<cv::Vec3b>(i_target, j_target)[2] : Lap.at<cv::Vec3f>(i_target, j_target)[2];

                Index_in_Target.push_back({ i_target,j_target });
                Index2Point[index] = { i,j };
                Point2Index[{i, j}] = index;
                index++;
            }
        }
    }
    
    // Calculate A
    cv::Mat A = cv::Mat::zeros(Dim, Dim, CV_32FC1);

    for (int i = 0; i < Dim; i++)
    {
        int x = Index2Point[i].first;
        int y = Index2Point[i].second;
        if (Mask_Boundary.at<float>(x, y))
        {
            A.at<float>(i, i) = 1;
        }
        else
        {
            std::map<std::pair<int, int>, int>::iterator up, down, left, right;
            try
            {
                up = Point2Index.find({ x - 1,y });
                down = Point2Index.find({ x + 1,y });
                left = Point2Index.find({ x ,y - 1 });
                right = Point2Index.find({ x ,y + 1 });
                if (up == Point2Index.end() ||
                    down == Point2Index.end() ||
                    left == Point2Index.end() ||
                    right == Point2Index.end())
                {
                    error = 3;
                    throw error;
                }
            }
            catch (int)
            {
                if (error == 3)
                    std::cout << "Failed to build mat A!" << std::endl;
                return error;
            }
            A.at<float>(i, up->second) = 1;
            A.at<float>(i, down->second) = 1;
            A.at<float>(i, left->second) = 1;
            A.at<float>(i, right->second) = 1;
            A.at<float>(i, i) = -4;
        }
    }

    // Calculate Ax=b
    cv::Mat x_answer(Dim, 1, CV_32FC3, cv::Scalar(0, 0, 0));
    cv::Mat x_b(Dim, 1, CV_32FC1, cv::Scalar(0));
    cv::Mat x_g(Dim, 1, CV_32FC1, cv::Scalar(0));
    cv::Mat x_r(Dim, 1, CV_32FC1, cv::Scalar(0));
    std::vector<cv::Mat>b_bgr;
    std::vector<cv::Mat>x_bgr;
    cv::split(b, b_bgr);
    cv::solve(A, b_bgr[0], x_b);
    cv::solve(A, b_bgr[1], x_g);
    cv::solve(A, b_bgr[2], x_r);
    x_bgr.push_back(x_b);
    x_bgr.push_back(x_g);
    x_bgr.push_back(x_r);
    cv::merge(x_bgr, x_answer);

    // Draw 
    for (int i = 0; i < Dim; i++)
    {
        int x = Index_in_Target[i].first;
        int y = Index_in_Target[i].second;
        if (x_answer.at<cv::Vec3f>(i, 0)[0] < 0)
        {
            x_answer.at<cv::Vec3f>(i, 0)[0] = 0;
        }
        else if (x_answer.at<cv::Vec3f>(i, 0)[0] > 255)
        {
            x_answer.at<cv::Vec3f>(i, 0)[0] = 255;
        }
        if (x_answer.at<cv::Vec3f>(i, 0)[1] < 0)
        {
            x_answer.at<cv::Vec3f>(i, 0)[1] = 0;
        }
        else if (x_answer.at<cv::Vec3f>(i, 0)[1] > 255)
        {
            x_answer.at<cv::Vec3f>(i, 0)[1] = 255;
        }
        if (x_answer.at<cv::Vec3f>(i, 0)[2] < 0)
        {
            x_answer.at<cv::Vec3f>(i, 0)[2] = 0;
        }
        else if (x_answer.at<cv::Vec3f>(i, 0)[2] > 255)
        {
            x_answer.at<cv::Vec3f>(i, 0)[2] = 255;
        }

        Target.at<cv::Vec3b>(x, y)[0] = x_answer.at<cv::Vec3f>(i, 0)[0];
        Target.at<cv::Vec3b>(x, y)[1] = x_answer.at<cv::Vec3f>(i, 0)[1];
        Target.at<cv::Vec3b>(x, y)[2] = x_answer.at<cv::Vec3f>(i, 0)[2];
    }
    cv::imshow("Result", Target);

    cv::imwrite("Result.png", Target);
    cv::waitKey(0);
   
    return 0;
}