#include "utils/logger.h"

Logger::Logger(const std::string &save_dir, const std::string &save_log_name, const std::string &log_title)
{
    std::cout << "[Logger] start to init logger: " << save_dir << std::endl;
    logger_ = spdlog::daily_logger_mt(log_title, save_dir + "/" + save_log_name, 0, 0);
    spdlog::set_default_logger(logger_);
    spdlog::flush_every(std::chrono::seconds(1));
    std::cout << "[Logger] init logger success..." << std::endl;
}

void Logger::info(const std::string &message)
{
    spdlog::info(message);
}

void Logger::error(const std::string &message)
{
    spdlog::error(message);
}

void Logger::warning(const std::string &message)
{
    spdlog::warn(message);
}

Logger::~Logger()
{
    spdlog::drop_all();
}
