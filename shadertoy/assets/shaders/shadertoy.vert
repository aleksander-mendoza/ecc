#version 450 core
//in vec3 aPos;

layout (location = 0) out vec2 fragCoord; // specify a color output to the fragment shader

void main()
{
    const vec2[6] vertices = vec2[6](
        // first triangle
        vec2(1.f,  1.f), // top right
        vec2(1.f, -1.f),   // bottom right
        vec2(-1.f,  1.f),   // top left
        // second triangle
        vec2(1.f, -1.f),   // bottom right
        vec2(-1.f, -1.f),   // bottom left
        vec2(-1.f,  1.f)   // top left
    );
    fragCoord = vertices[gl_VertexIndex];
    gl_Position = vec4(vertices[gl_VertexIndex], 0.0, 1.0);
}