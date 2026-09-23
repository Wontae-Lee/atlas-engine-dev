/**
 * @file
 * @brief Defines commands accepted by the interactive server.
 */

#pragma once

namespace atlas::interactive {

/// Identifies an operation accepted by the interactive server.
enum class Command {
    create,   ///< Create a session from a simulation configuration.
    start,    ///< Put a session into continuous-running state.
    pause,    ///< Pause automatic session updates.
    step,     ///< Advance a session by an explicit number of steps.
    status,   ///< Query the current session status.
    save,     ///< Serialize the current core system state.
    restart,  ///< Recreate a session from its original configuration.
    close,    ///< Destroy one session.
    shutdown  ///< Request termination of the interactive server.
};

}
