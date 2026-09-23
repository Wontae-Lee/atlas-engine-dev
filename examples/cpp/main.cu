/**
 * @file
 * @brief Selects and runs one native Atlas example case.
 */

#include "cases/cylinder.h"
#include "template.h"

#include <cstddef>
#include <cstdio>
#include <cstdlib>
#include <string_view>

/// Parses the selected case and requested step count, then runs the example.
int
main(int argc, char** argv) {
    // Case selection keeps one executable usable as more focused examples are added.
    const std::string_view case_name = argc > 1 ? argv[1] : "cylinder";
    const std::size_t steps = argc > 2 ? std::strtoul(argv[2], nullptr, 10) : 40;

    if (case_name == "cylinder") return atlas_examples::cylinder::run(steps);

    std::fprintf(stderr, "unknown example case: %.*s\n",
                 static_cast<int>(case_name.size()), case_name.data());
    return 1;
}
