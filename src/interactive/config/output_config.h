/**
 * @file
 * @brief Defines application-owned output settings for an interactive session.
 */

#pragma once

#include <filesystem>
#include <string>

namespace atlas::interactive {

/**
 * @brief Selects optional application-level output for a simulation session.
 *
 * Output policy remains outside Atlas core. A session reads this value when it
 * configures reporting for an interactive client.
 */
struct OutputConfig {
    bool csv_enabled = false;                    ///< Enables periodic CSV statistics output.
    std::filesystem::path output_directory = "."; ///< Directory that receives output files.
    std::string csv_filename = "statistics.csv"; ///< Name of the statistics CSV file.
};

}
