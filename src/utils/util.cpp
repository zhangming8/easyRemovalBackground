
#include "utils/util.h"

template <typename AnyCls>
std::ostream &operator<<(std::ostream &os, const std::vector<AnyCls> &v)
{
    os << "{";
    for (auto it = v.begin(); it != v.end(); ++it)
    {
        os << "(" << *it << ")";
        if (it != v.end() - 1)
            os << ", ";
    }
    os << "}";
    return os;
}

std::chrono::system_clock::time_point time_now()
{
    auto x = std::chrono::system_clock::now();
    return x;
}

double time_cost(const std::chrono::system_clock::time_point &t2, const std::chrono::system_clock::time_point &t1)
{
    double dr = std::chrono::duration<double>(t2 - t1).count();
    // double dr = std::chrono::duration<double, std::milli>(t2 - t1).count();
    // double dr = std::chrono::duration<double, std::micro>(t2 - t1).count();
    // double dr = std::chrono::duration<double, std::nano>(t2 - t1).count();
    return dr;
}

bool ends_with(const std::string &str, const std::string &suffix)
{
    if (str.length() < suffix.length())
    {
        return false;
    }

    std::string str_suffix = str.substr(str.length() - suffix.length());
    return str_suffix.compare(suffix) == 0;
}

bool file_exist(const std::string &filepath)
{
    std::ifstream f(filepath.c_str());
    return f.good();
}

std::string lower_str(std::string str)
{
    std::string res;
    for (char c : str)
    {
        char s = std::tolower(c);
        res += std::string(1, s);
    }
    return res;
}

void replace_str(std::string &str, const std::string &src, const std::string &dst)
{
    size_t pos = 0;
    while ((pos = str.find(src, pos)) != std::string::npos)
    {
        str.replace(pos, src.length(), dst);
        pos += dst.length();
    }
}

std::string base_name(const std::string &path)
{
    std::string path2 = path;
    replace_str(path2, "\\", "/");
    size_t pos = path2.find_last_of("/");

    std::string basename;
    if (pos != std::string::npos)
        basename = path2.substr(pos + 1);
    else
        basename = path2;
    return basename;
}

std::string dir_name(const std::string &path)
{
    std::string path2 = path;
    replace_str(path2, "\\", "/");
    size_t pos = path2.find_last_of("/");

    std::string dirname = "";
    if (pos != std::string::npos)
        dirname = path2.substr(0, pos);
    else
        dirname = path2;
    return dirname;
}

void print_mat(const cv::Mat &mat)
{
    std::cout << "--------------- h: " << mat.rows << ", w: " << mat.cols << ", type: " << mat.type() << std::endl;

    for (int i = 0; i < mat.rows; i++)
    {
        for (int j = 0; j < mat.cols; j++)
        {
            if (6 == mat.type())
                std::cout << mat.at<double>(i, j) << " ";
            else if (5 == mat.type())
                std::cout << mat.at<float>(i, j) << " ";
            else
            {
                std::cout << "unknown mat type: " << mat.type() << std::endl;
                std::cout << mat.at<float>(i, j) << " ";
            }
        }
        std::cout << std::endl;
    }
}

std::string print_1d_vector(const std::vector<int> &vec)
{
    std::string r = "[";
    for (size_t i = 0; i < vec.size(); ++i)
    {
        r += std::to_string(vec[i]);
        if (i != vec.size() - 1)
            r += ", ";
    }
    r += "]";
    return r;
}

std::string print_1d_vector(const std::vector<long long int> &vec)
{
    std::string r = "[";
    for (size_t i = 0; i < vec.size(); ++i)
    {
        r += std::to_string(vec[i]);
        if (i != vec.size() - 1)
            r += ", ";
    }
    r += "]";
    return r;
}

std::string print_1d_vector(const std::set<long long int> &vec)
{
    std::string r = "(";
    for (auto it = vec.begin(); it != vec.end(); ++it)
    {
        r += std::to_string(*it);
        if (it != --vec.end())
            r += ", ";
    }
    r += ")";
    return r;
}

std::string print_1d_vector(const std::vector<float> &vec)
{
    std::string r = "[";
    for (size_t i = 0; i < vec.size(); ++i)
    {
        r += std::to_string(vec[i]);
        if (i != vec.size() - 1)
            r += ", ";
    }
    r += "]";
    return r;
}

std::string print_1d_vector(const std::vector<std::string> &vec)
{
    std::string r = "[";
    for (size_t i = 0; i < vec.size(); ++i)
    {
        r += ("'" + vec[i] + "'");
        if (i != vec.size() - 1)
            r += ", ";
    }
    r += "]";
    return r;
}

std::string print_map2(const std::map<int, std::vector<int>> &dict)
{
    std::string result = "{";
    for (auto it = dict.begin(); it != dict.end(); ++it)
    {
        result += std::to_string(it->first) + ": [";
        const std::vector<int> &values = it->second;
        for (auto vect_it = values.begin(); vect_it != values.end(); ++vect_it)
        {
            result += std::to_string(*vect_it);
            if (std::next(vect_it) != values.end())
                result += ", ";
        }
        result += "]";
        if (std::next(it) != dict.end())
            result += ", ";
    }
    result += "}";
    return result;
}

std::string repeat_string(const std::string &str, int n)
{
    std::string result;
    for (int i = 0; i < n; ++i)
    {
        result += str;
    }
    return result;
}

std::string print_vector_vector_pt(const std::vector<std::vector<cv::Point>> &polys)
{
    std::stringstream ss;
    ss << "[";
    for (const auto &poly : polys)
    {
        ss << "[";
        for (const auto &point : poly)
        {
            ss << "(" << point.x << ", " << point.y << ") ";
        }
        ss << "], ";
    }
    ss << "]";
    return ss.str();
}

std::string print_set_int(const std::set<int> &sets)
{
    std::stringstream ss;
    ss << "[";
    auto it = sets.begin();
    if (it != sets.end())
    {
        ss << *it;
        ++it;
    }
    for (; it != sets.end(); ++it)
    {
        ss << ", " << *it;
    }
    ss << "]";
    return ss.str();
}

std::string get_current_str_time()
{
    std::chrono::system_clock::time_point now = std::chrono::system_clock::now();
    std::time_t timestamp = std::chrono::system_clock::to_time_t(now);
    std::tm *localTime = std::localtime(&timestamp);

    std::ostringstream oss;
    oss << std::put_time(localTime, "%Y-%m-%d_%H-%M-%S");
    // oss << std::put_time(localTime, "%H-%M-%S");

    return oss.str();
}

std::vector<std::string> read_txt(std::string &txt_file)
{
    std::vector<std::string> data;
    if (!file_exist(txt_file))
    {
        std::cout << "txt_file not exist: " << txt_file << std::endl;
        return data;
    }

    std::ifstream file(txt_file);
    if (file.is_open())
    {
        std::string line;
        while (std::getline(file, line))
            data.emplace_back(line);
        file.close();
    }
    else
        std::cout << "cannot open file: " << txt_file << std::endl;
    return data;
}

std::vector<std::string> split_str(const std::string &s, const std::string &delimiters)
{
    std::vector<std::string> splits;
    size_t last_pos = s.find_first_not_of(delimiters, 0);
    size_t pos = s.find(delimiters, last_pos);
    while (last_pos != std::string::npos)
    {
        splits.emplace_back(s.substr(last_pos, pos - last_pos));
        last_pos = s.find_first_not_of(delimiters, pos);
        pos = s.find(delimiters, last_pos);
    }
    return splits;
}

std::string join_str(const std::vector<std::string> &splits, const std::string &delimiter)
{
    std::string join_s = "";
    for (int i = 0; i < splits.size(); ++i)
    {
        join_s += splits[i];
        if (i != splits.size() - 1)
            join_s += delimiter;
    }
    return join_s;
}

std::vector<std::string> list_folder(const std::string folder_path)
{
    std::vector<std::string> folders;
    for (const auto &entry : std::filesystem::directory_iterator(folder_path))
    {
        if (entry.is_directory())
            folders.emplace_back(entry.path().string());
    }
    return folders;
}

bool str1_in_str2(const std::string &str1, const std::string &str2)
{
    return str2.find(str1) != std::string::npos;
}

bool is_folder_exists(const std::string &path)
{
    struct stat info;
    return (stat(path.c_str(), &info) == 0 && (info.st_mode & S_IFDIR));
}

void mkdir(const std::string &dir)
{
    if (!std::filesystem::exists(dir))
    {
        if (std::filesystem::create_directory(dir))
        {
            // std::cout << "create folder success: " << dir << std::endl;
        }
        else
        {
            std::cerr << "create folder failed: " << dir << std::endl;
        }
    }
}

void makedirs(const std::string &path)
{
    size_t pos = 0;
    std::string current_path;

    while ((pos = path.find_first_of("/\\", pos)) != std::string::npos)
    {
        current_path = path.substr(0, pos++);
        auto tmp = lower_str(current_path);
        if (!is_folder_exists(current_path))
        {
            // std::cout << "not exists:" << current_path << std::endl;
            if (_mkdir(current_path.c_str()) != 0)
            {
                std::cerr << "Error creating directory: " << current_path << std::endl;
                return;
            }
            // std::cout << "Directory created: " << current_path << std::endl;
        }
    }

    if (!is_folder_exists(path))
    {
        if (_mkdir(path.c_str()) != 0)
        {
            std::cerr << "Error creating directory: " << path << std::endl;
        }
        else
        {
            // std::cout << "Directory created: " << path << std::endl;
        }
    }
    else
    {
        // std::cout << "Directory already exists: " << path << std::endl;
    }
}

std::string change_file_extend(const std::string &file_path, const std::string &new_extend)
{
    size_t pos = file_path.find_last_of(".");

    if (pos != std::string::npos)
    {
        return file_path.substr(0, pos) + "." + new_extend;
    }
    else
    {
        return file_path + "." + new_extend;
    }
}

std::string float_to_string(float num, int precision)
{
    std::ostringstream oss;
    oss << std::fixed << std::setprecision(precision) << num;
    return oss.str();
}

std::string print_mat_list(const cv::Mat &mat, int float_precision)
{
    std::ostringstream oss;
    oss << "[";

    for (int i = 0; i < mat.rows; ++i)
    {
        if (i > 0)
            oss << ", ";
        oss << "[";

        for (int j = 0; j < mat.cols; ++j)
        {
            if (j > 0)
                oss << ", ";

            if (mat.channels() > 1)
            {
                oss << "[";
                cv::Vec3b pixel = mat.at<cv::Vec3b>(i, j);
                for (int c = 0; c < mat.channels(); ++c)
                {
                    if (c > 0)
                        oss << ", ";
                    oss << static_cast<int>(pixel[c]);
                }
                oss << "]";
            }
            else
            {
                switch (mat.depth())
                {
                case CV_8U:
                    oss << static_cast<int>(mat.at<uchar>(i, j));
                    break;
                case CV_32F:
                    oss << std::fixed << std::setprecision(float_precision)
                        << mat.at<float>(i, j);
                    break;
                case CV_64F:
                    oss << std::fixed << std::setprecision(float_precision)
                        << mat.at<double>(i, j);
                    break;
                default:
                    oss << "?";
                }
            }
        }
        oss << "]";
    }
    oss << "]";
    return oss.str();
}

std::vector<std::string> get_devices()
{
    std::vector<std::string> device_names;
    int num_gpu;
    cudaGetDeviceCount(&num_gpu);

    for (int device = 0; device < num_gpu; device++)
    {
        cudaDeviceProp prop;
        cudaGetDeviceProperties(&prop, device);
        std::cout << "device name: " << prop.name << " major: " << prop.major << " minor: " << prop.minor << " async_engine_count: " << prop.asyncEngineCount << std::endl;

        device_names.push_back(std::string(prop.name));
    }
    return device_names;
}
