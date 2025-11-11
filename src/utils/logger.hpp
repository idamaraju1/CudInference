#pragma once

#include <iostream>
#include <sstream>
#include <string>
#include <chrono>
#include <iomanip>

namespace onnx_runner {

enum class LogLevel {
    DEBUG,
    INFO,
    WARNING,
    ERROR
};

class Logger {
public:
    static Logger& instance() {
        static Logger logger;
        return logger;
    }

    void setLevel(LogLevel level) { min_level_ = level; }

    template<typename... Args>
    void debug(Args&&... args) {
        log(LogLevel::DEBUG, std::forward<Args>(args)...);
    }

    template<typename... Args>
    void info(Args&&... args) {
        log(LogLevel::INFO, std::forward<Args>(args)...);
    }

    template<typename... Args>
    void warning(Args&&... args) {
        log(LogLevel::WARNING, std::forward<Args>(args)...);
    }

    template<typename... Args>
    void error(Args&&... args) {
        log(LogLevel::ERROR, std::forward<Args>(args)...);
    }

private:
    Logger() : min_level_(LogLevel::INFO) {}

    LogLevel min_level_;

    template<typename... Args>
    void log(LogLevel level, Args&&... args) {
        if (level < min_level_) return;

        std::ostringstream oss;
        oss << "[" << getTimestamp() << "] "
            << "[" << levelToString(level) << "] ";

        (oss << ... << args);

        if (level == LogLevel::ERROR) {
            std::cerr << oss.str() << std::endl;
        } else {
            std::cout << oss.str() << std::endl;
        }
    }

    std::string levelToString(LogLevel level) {
        switch (level) {
            case LogLevel::DEBUG:   return "DEBUG";
            case LogLevel::INFO:    return "INFO ";
            case LogLevel::WARNING: return "WARN ";
            case LogLevel::ERROR:   return "ERROR";
            default: return "UNKNOWN";
        }
    }

    std::string getTimestamp() {
        auto now = std::chrono::system_clock::now();
        auto time = std::chrono::system_clock::to_time_t(now);
        auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(
            now.time_since_epoch()) % 1000;

        std::stringstream ss;
        ss << std::put_time(std::localtime(&time), "%H:%M:%S");
        ss << '.' << std::setfill('0') << std::setw(3) << ms.count();
        return ss.str();
    }
};

// Convenience macros
#define LOG_DEBUG(...) onnx_runner::Logger::instance().debug(__VA_ARGS__)
#define LOG_INFO(...)  onnx_runner::Logger::instance().info(__VA_ARGS__)
#define LOG_WARN(...)  onnx_runner::Logger::instance().warning(__VA_ARGS__)
#define LOG_ERROR(...) onnx_runner::Logger::instance().error(__VA_ARGS__)

} // namespace onnx_runner
