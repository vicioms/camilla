#pragma once
#include <string>
#include <unordered_map>
#include <vector>
#include <stdexcept>
#include <sstream>

class ArgParser {
public:
    std::unordered_map<std::string, std::string> options;
    std::vector<std::string> positional;
    
    ArgParser(int argc, char** argv) {
        for (int i = 1; i < argc; ++i) {
            std::string arg = argv[i];
            if (arg.rfind("--", 0) == 0) {  // starts with --
                std::string key = arg.substr(2);
                std::string value = "true"; // default for flag
                if (i + 1 < argc && std::string(argv[i + 1]).rfind("--", 0) != 0) {
                    value = argv[++i];  // consume next token
                }
                options[key] = value;
            } else {
                positional.push_back(arg);
            }
        }
    }

    bool has(const std::string& key) const {
        return options.find(key) != options.end();
    }

    std::string get(const std::string& key, const std::string& default_val = "") const {
        auto it = options.find(key);
        return it != options.end() ? it->second : default_val;
    }

    int get_int(const std::string& key, int default_val = 0) const {
        auto it = options.find(key);
        if (it != options.end()) return std::stoi(it->second);
        return default_val;
    }

    float get_float(const std::string& key, float default_val = 0.0f) const {
        auto it = options.find(key);
        if (it != options.end()) return std::stof(it->second);
        return default_val;
    }

    bool get_bool(const std::string& key) const {
        auto it = options.find(key);
        return it != options.end() && (it->second == "true" || it->second == "1");
    }
};
