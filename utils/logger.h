#pragma once

#include <string>

namespace logger {

enum Level { INFO = 0, WARN = 1, ERROR = 2, DEBUG = 3 };

void setLevel(Level level);
Level getLevel();

#ifdef _WIN32
void addOutputFile(const std::wstring& path);
#else
void addOutputFile(const std::string& path);
#endif

void closeOutputFile();

void info(const char* fmt, ...);
void warn(const char* fmt, ...);
void error(const char* fmt, ...);
void debug(const char* fmt, ...);

}  // namespace logger

#define INFO(fmt, ...) logger::info(fmt, ##__VA_ARGS__)
#define WARN(fmt, ...) logger::warn(fmt, ##__VA_ARGS__)
#define ERROR(fmt, ...) logger::error(fmt, ##__VA_ARGS__)
#define DEBUG(fmt, ...) logger::debug(fmt, ##__VA_ARGS__)