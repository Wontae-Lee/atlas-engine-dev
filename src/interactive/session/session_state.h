/**
 * @file
 * @brief Defines externally reportable simulation-session states.
 */

#pragma once

namespace atlas::interactive {

/// Lifecycle state of an initialized interactive session.
enum class SessionState {
    ready,   ///< Initialized and waiting for its first command.
    running, ///< Advanced once whenever Server::update is called.
    paused   ///< Initialized but excluded from automatic updates.
};

}
