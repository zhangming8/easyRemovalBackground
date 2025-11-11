#pragma once
#include <chrono>
#include <iostream>
#include <sstream>
#include <fstream>
#include <filesystem>
#include <direct.h>

#include <cudnn.h>
#include <cuda_runtime.h>

#include <opencv2/opencv.hpp>
#include "utils/logger.h"

struct Args
{
    std::string device = "", app_version = "", app_name = "", email = "", website = "";
    std::string root_path = "";
};

struct ObjectMatting
{
    cv::Mat alpha;
};

template <typename AnyCls>
std::ostream &operator<<(std::ostream &os, const std::vector<AnyCls> &v);

std::chrono::system_clock::time_point time_now();
double time_cost(const std::chrono::system_clock::time_point &t2, const std::chrono::system_clock::time_point &t1);
bool ends_with(const std::string &str, const std::string &suffix);
bool file_exist(const std::string &filepath);
std::string lower_str(std::string str);
void print_mat(const cv::Mat &mat);
std::string print_1d_vector(const std::vector<float> &vec);
std::string print_1d_vector(const std::vector<int> &vec);
std::string print_1d_vector(const std::vector<long long int> &vec);
std::string print_1d_vector(const std::set<long long int> &vec);
std::string print_1d_vector(const std::vector<std::string> &vec);
std::string print_map2(const std::map<int, std::vector<int>> &dict);
std::string repeat_string(const std::string &str, int n);
std::string print_vector_vector_pt(const std::vector<std::vector<cv::Point>> &polys);
std::string print_set_int(const std::set<int> &sets);
void replace_str(std::string &str, const std::string &src, const std::string &dst);
std::string base_name(const std::string &path);
std::string dir_name(const std::string &path);
std::string get_current_str_time();
std::vector<std::string> read_txt(std::string &txt_file);
std::vector<std::string> split_str(const std::string &s, const std::string &delimiters);
std::string join_str(const std::vector<std::string> &splits, const std::string &delimiter);
std::vector<std::string> list_folder(const std::string folder_path);
bool str1_in_str2(const std::string &str1, const std::string &str2);
bool is_folder_exists(const std::string &path);
void mkdir(const std::string &dir);
void makedirs(const std::string &path);
std::string change_file_extend(const std::string &file_path, const std::string &new_extend);
std::string float_to_string(float num, int precision = 2);
std::string print_mat_list(const cv::Mat &mat, int float_precision = 6);
std::vector<std::string> get_devices();
