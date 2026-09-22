#pragma once

#include <filesystem>
#include <string>

namespace atlas::interactive {

struct SessionConfig final {
    bool csv_enabled = false;
    std::filesystem::path output_directory = ".";
    std::string csv_filename = "statistics.csv";
};

}
