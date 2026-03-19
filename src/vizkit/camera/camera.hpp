#pragma once

#ifdef ATLAS_ENABLE_VIZKIT

namespace atlas::vizkit {
void
Camera::init_mouse_controls(GLFWwindow* w) {
    if (mouse_initialized || w == nullptr) return;

    glfwSetWindowUserPointer(w, this);
    glfwSetScrollCallback(w, &Camera::scroll_callback);
    glfwGetCursorPos(w, &last_cursor_x, &last_cursor_y);
    mouse_initialized = true;
}

void
Camera::scroll_callback(GLFWwindow* w, double xoffset, double yoffset) {
    (void)xoffset;

    if (w == nullptr) return;

    auto* camera = static_cast<Camera*>(glfwGetWindowUserPointer(w));
    if (camera == nullptr) return;

    camera->pending_scroll_zoom -= static_cast<float>(yoffset) * 0.8f;
}

void
Camera::handle(GLFWwindow* w) {
    init_mouse_controls(w);

    // Orbit camera state is represented with three scalars:
    // - yaw   : rotation around the global Z axis (azimuth)
    // - pitch : elevation angle measured from the XY plane
    // - dist  : distance from the camera target to the eye point
    //
    // This is effectively a spherical-coordinate parameterization of the eye
    // position relative to a fixed look-at center.
    double cursor_x = last_cursor_x;
    double cursor_y = last_cursor_y;
    glfwGetCursorPos(w, &cursor_x, &cursor_y);

    const float delta_x = static_cast<float>(cursor_x - last_cursor_x);
    const float delta_y = static_cast<float>(cursor_y - last_cursor_y);

    const bool left_pressed = glfwGetMouseButton(w, GLFW_MOUSE_BUTTON_LEFT) == GLFW_PRESS;
    if (left_pressed) {
        if (left_drag_active) {
            yaw -= delta_x * 0.008f;
            pitch -= delta_y * 0.008f;
        }
        left_drag_active = true;
    } else {
        left_drag_active = false;
    }

    const bool right_pressed = glfwGetMouseButton(w, GLFW_MOUSE_BUTTON_RIGHT) == GLFW_PRESS;
    if (right_pressed) {
        if (right_drag_active) {
            dist += delta_y * 0.05f;
        }
        right_drag_active = true;
    } else {
        right_drag_active = false;
    }

    last_cursor_x = cursor_x;
    last_cursor_y = cursor_y;

    // Horizontal orbit:
    // pressing A/D changes the azimuth angle, so the camera moves on a circle
    // around the target when projected onto the XY plane.
    if (glfwGetKey(w, GLFW_KEY_A) == GLFW_PRESS) yaw -= 0.02f;
    if (glfwGetKey(w, GLFW_KEY_D) == GLFW_PRESS) yaw += 0.02f;

    // Vertical orbit:
    // pressing W/S changes the elevation angle. Positive pitch lifts the eye
    // upward, negative pitch lowers it below the target's horizontal plane.
    if (glfwGetKey(w, GLFW_KEY_W) == GLFW_PRESS) pitch += 0.02f;
    if (glfwGetKey(w, GLFW_KEY_S) == GLFW_PRESS) pitch -= 0.02f;

    // Radial zoom:
    // pressing Q/E changes only the orbit radius. The view direction still
    // points at the same target, but the eye moves farther away or closer.
    if (glfwGetKey(w, GLFW_KEY_Q) == GLFW_PRESS) dist += 0.05f;
    if (glfwGetKey(w, GLFW_KEY_E) == GLFW_PRESS) dist -= 0.05f;
    dist += pending_scroll_zoom;
    pending_scroll_zoom = 0.0f;

    // Clamp pitch so the forward vector does not become nearly parallel to the
    // world up vector.
    //
    // Why this matters mathematically:
    // the view basis later uses
    //   s = normalize(cross(fwd, up))
    // If fwd and up are parallel or almost parallel, then
    //   |cross(fwd, up)| = |fwd||up|sin(theta)
    // becomes 0 or numerically tiny because theta -> 0 or pi.
    // That makes normalization unstable and the camera basis ill-defined.
    //
    // The chosen bound 1.2 rad (~68.8 deg) keeps enough angular separation from
    // the pole while still allowing a high viewing angle.
    pitch = fminf(fmaxf(pitch, -1.2f), 1.2f);

    // Clamp the orbit radius to keep the camera usable.
    //
    // A very small distance would place the eye almost on top of the target,
    // amplifying tiny angular changes into abrupt screen motion.
    // A very large distance would collapse perspective cues and make the scene
    // visually tiny. These bounds enforce a practical interaction range.
    dist = fminf(fmaxf(dist, 2.0f), 100.0f);
}

void
Camera::build_mvp(int w, int h, float out_mvp[16]) const {
    {
        // Aspect ratio of the viewport.
        //
        // In a perspective projection, horizontal scaling depends on width /
        // height so that a unit square in normalized device coordinates maps
        // correctly onto a non-square window.
        const float aspect = (h > 0) ? static_cast<float>(w) / static_cast<float>(h) : 1.0f;

        // Convert a 45-degree vertical field of view to radians because the
        // trigonometric functions below operate in radians.
        const float fov = 45.0f * static_cast<float>(M_PI) / 180.0f;

        // Standard perspective scale factor:
        //   f = 1 / tan(fov / 2) = cot(fov / 2)
        //
        // Geometric interpretation:
        // consider the half-height of the near image plane at unit distance.
        // If that half-height is tan(fov/2), then dividing by it normalizes the
        // vertical extent so points on the frustum boundary map to clip-space
        // limits after projection.
        const float f = 1.0f / tanf(fov * 0.5f);

        // Perspective projection matrix in column-major storage.
        //
        // Written as a mathematical 4x4 matrix, this corresponds to:
        //
        //   [ f/aspect   0    0      0 ]
        //   [    0       f    0      0 ]
        //   [    0       0   -1   -0.2 ]
        //   [    0       0   -1      0 ]
        //
        // The first two diagonal terms scale x and y according to the field of
        // view and aspect ratio.
        //
        // The last two rows are the projective part:
        // - z contributes to w through the -1 term, so after homogeneous divide
        //   we obtain the familiar perspective effect where farther objects
        //   appear smaller.
        // - the -0.2 term shifts depth into clip space.
        //
        // The matrix is intentionally stored as a flat array in column-major
        // order, matching common OpenGL-style upload conventions.
        const float proj[16] = { f / aspect, 0, 0, 0, 0, f, 0, 0, 0, 0, -1, -1, 0, 0, -0.2f, 0 };

        // Convert orbit parameters to a Cartesian eye position.
        //
        // Using spherical coordinates with radius r = dist:
        //   x = r cos(pitch) cos(yaw)
        //   y = r cos(pitch) sin(yaw)
        //   z = r sin(pitch)
        //
        // Explanation:
        // - cos(pitch) is the projection of the radius onto the XY plane
        // - multiplying by cos(yaw), sin(yaw) splits that planar radius into X
        //   and Y components
        // - sin(pitch) gives the vertical Z component
        const float ex = dist * cosf(pitch) * cosf(yaw);
        const float ey = dist * cosf(pitch) * sinf(yaw);
        const float ez = dist * sinf(pitch);

        // The eye uses (-ex, -ey, ez), not (ex, ey, ez).
        //
        // That sign choice fixes the screen-space orbit convention so the
        // camera appears to move in the intuitive direction when yaw changes.
        // The camera always looks at a fixed point slightly above the origin.
        //
        // center = (0, 0, 0.6) means the view is biased upward a bit instead of
        // focusing exactly on the ground-plane origin. This is often useful in
        // visualization because objects of interest tend to occupy positive Z.
        const Vector3F eye { -ex, -ey, ez }, center { 0, 0, 0.6f }, up { 0, 0, 1 };

        // Build a right-handed orthonormal camera frame.
        //
        // 1. Forward direction:
        //      fwd = normalize(center - eye)
        //    This is the unit vector pointing from the eye toward the target.
        //
        // 2. Right direction:
        //      s = normalize(cross(fwd, up))
        //    The cross product of forward and world-up yields a vector
        //    perpendicular to both, i.e. the camera's right axis.
        //
        // 3. Corrected up direction:
        //      u = cross(s, fwd)
        //    This recomputes an up vector guaranteed to be orthogonal to both
        //    right and forward, producing an orthonormal basis.
        //
        // This is the standard "look-at" construction. The normalization steps
        // make the basis vectors unit length, which prevents unintended scaling
        // in the view transform.
        const Vector3F fwd = normalize(center - eye);
        const Vector3F s   = normalize(cross(fwd, up));
        const Vector3F u   = cross(s, fwd);

        // View matrix in column-major storage.
        //
        // In matrix form, the rigid transform maps a world-space point p into
        // camera space by:
        //   p_camera = R * (p_world - eye)
        //
        // Expanding this into a homogeneous 4x4 matrix gives:
        //
        //   [  s.x    s.y    s.z   -dot(s, eye) ]
        //   [  u.x    u.y    u.z   -dot(u, eye) ]
        //   [ -f.x   -f.y   -f.z    dot(f, eye) ]
        //   [   0      0      0          1      ]
        //
        // The minus sign on forward is conventional in many graphics APIs:
        // the camera looks down its negative Z axis in view space.
        //
        // The translation terms are not arbitrary constants. They come from
        // projecting the eye point onto the camera basis so that the camera
        // itself becomes the origin after transformation.
        const float view[16] = { s.x, u.x, -fwd.x, 0, s.y, u.y, -fwd.y, 0, s.z, u.z, -fwd.z, 0, -dot(s, eye), -dot(u, eye), dot(fwd, eye), 1 };

        // Compute the combined matrix:
        //   MVP = Projection * View
        //
        // If vectors are treated as column vectors, multiplication order is
        // right-to-left:
        //   p_clip = MVP * p_world = Projection * (View * p_world)
        //
        // Storage detail:
        // - element at row r, column c is stored at index c*4 + r
        // - this is column-major linearization
        //
        // Algebraically, matrix multiplication is:
        //   out[r, c] = sum_k proj[r, k] * view[k, c]
        //
        // The indexing below implements exactly that formula under the chosen
        // storage convention.
        for (int c = 0; c < 4; ++c)
            for (int r = 0; r < 4; ++r) {
                // Start the dot product for one output matrix element.
                out_mvp[c * 4 + r] = 0;

                // Accumulate the dot product between:
                // - row r of the projection matrix
                // - column c of the view matrix
                for (int k = 0; k < 4; ++k)
                    out_mvp[c * 4 + r] += proj[k * 4 + r] * view[c * 4 + k];
            }
    }
}
}

#endif
