#pragma once

#ifdef ATLAS_ENABLE_VIZKIT
namespace atlas::vizkit {
static auto k_point_vs = R"(
#version 330 core
layout(location=0) in vec3 aPos;
uniform mat4 MVP;
uniform float uPointSize;
void main(){ gl_Position = MVP * vec4(aPos,1.0); gl_PointSize=uPointSize; }
)";
static auto k_point_fs = R"(
#version 330 core
uniform vec4 uColor;
out vec4 FragColor;
void main(){
    vec2 c = gl_PointCoord*2.0-1.0;
    if(dot(c,c)>1.0) discard;
    FragColor = uColor;
}
)";
static auto k_point_color_vs = R"(
#version 330 core
layout(location=0) in vec3 aPos;
layout(location=1) in vec4 aColor;
uniform mat4 MVP;
uniform float uPointSize;
out vec4 vColor;
void main(){ gl_Position = MVP * vec4(aPos,1.0); gl_PointSize=uPointSize; vColor=aColor; }
)";
static auto k_point_color_fs = R"(
#version 330 core
in vec4 vColor;
out vec4 FragColor;
void main(){
    vec2 c = gl_PointCoord*2.0-1.0;
    if(dot(c,c)>1.0) discard;
    FragColor = vColor;
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
