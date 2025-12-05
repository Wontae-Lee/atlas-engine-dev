
#pragma once

#ifdef ATLAS_ENABLE_VIZKIT
namespace atlas::vizkit {
static auto k_point_vs = R"(
#version 330 core
layout(location=0) in vec3 aPos;
uniform mat4 MVP;
void main(){ gl_Position = MVP * vec4(aPos,1.0); gl_PointSize=4.0; }
)";
static auto k_point_fs = R"(
#version 330 core
out vec4 FragColor;
void main(){
    vec2 c = gl_PointCoord*2.0-1.0;
    if(dot(c,c)>1.0) discard;
    FragColor = vec4(0.0,0.0,1.0,0.5);
}
)";
static auto k_line_vs  = R"(
#version 330 core
layout(location=0) in vec3 aPos;
uniform mat4 MVP;
void main(){ gl_Position = MVP * vec4(aPos,1.0); }
)";
static auto k_line_fs  = R"(
#version 330 core
uniform vec4 uColor;
out vec4 FragColor;
void main(){ FragColor = uColor; }
)";
}

#endif
