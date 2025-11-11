#pragma once
#include <iostream>
#include <filesystem>

#include "spdlog/sinks/daily_file_sink.h"
#include "spdlog/spdlog.h"

class Logger
{
public:
   static Logger &GetInstance(const std::string &save_dir, const std::string &save_log_name, const std::string &log_title)
   {
      static Logger logger(save_dir + "/logs", save_log_name, log_title);
      return logger;
   }
   std::shared_ptr<spdlog::logger> get_logger() { return logger_; }
   Logger(Logger const &) = delete;            // Copy construct
   Logger(Logger &&) = delete;                 // Move construct
   Logger &operator=(Logger const &) = delete; // Copy assign
   Logger &operator=(Logger &&) = delete;      // Move assign

public:
   void info(const std::string &message);
   void error(const std::string &message);
   void warning(const std::string &message);

private:
   Logger(const std::string &save_dir, const std::string &save_log_name, const std::string &log_title = "daily_logger");
   ~Logger();
   std::shared_ptr<spdlog::logger> logger_;
};
