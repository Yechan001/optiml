/**
 * @file console.h
 * @brief Console input/output utilities
 * 
 * Provides cross-platform console handling with color support
 * and advanced display capabilities.
 */

#pragma once

#include <string>

namespace console {
    /**
     * @brief Console display modes
     * 
     * Controls how text is displayed in the console,
     * including color and formatting.
     */
    enum display_t {
        reset = 0,    ///< Reset to default display
        prompt,       ///< Prompt display style
        user_input,   ///< User input display style  
        error         ///< Error message display style
    };

    /**
     * @brief Initialize console subsystem
     * 
     * @param use_simple_io Use simplified I/O mode (better for subprocesses)
     * @param use_advanced_display Enable advanced display features like colors
     */
    void init(bool use_simple_io, bool use_advanced_display);

    /**
     * @brief Clean up console resources
     */
    void cleanup();

    /**
     * @brief Set current display mode
     * 
     * @param display Display mode to use
     */
    void set_display(display_t display);

    /**
     * @brief Read a line of input from console
     * 
     * @param line Output string for the read line
     * @param multiline_input Whether to allow multiline input
     * @return true if line was read successfully, false on EOF/error
     */
    bool readline(std::string & line, bool multiline_input);
}
