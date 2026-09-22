#pragma once

#include <filesystem>
#include <string>

namespace atlas::interactive {

struct OutputConfig {
    bool csv_enabled = false;
    std::filesystem::path output_directory = ".";
    std::string csv_filename = "statistics.csv";
};

}
