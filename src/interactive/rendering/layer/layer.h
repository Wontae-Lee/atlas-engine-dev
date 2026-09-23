/**
 * @file
 * @brief Declares the common interface for renderer layers.
 */

#pragma once

namespace atlas::interactive {

class Camera;
struct RenderState;

/// Draws one independent aspect of a prepared RenderState.
class Layer {
public:
    virtual ~Layer() = default;
    /// Creates OpenGL resources while a valid context is current.
    virtual void initialize() = 0;
    /// Draws the layer into the active render target.
    virtual void render(const RenderState& state,
                        const Camera& camera,
                        float aspect_ratio) = 0;
    /// Releases OpenGL resources while a valid context is current.
    virtual void shutdown() = 0;
};

}
