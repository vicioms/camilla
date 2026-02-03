#include "geometry.cuh"
#include "utils.cuh"
#include <string>
#include <iostream>
#include <fstream>
#include <string>
#include <map>
//#include <GL/glew.h>
//#include <cuda_gl_interop.h>
//#include <glm/glm.hpp>
//#include <glm/gtc/matrix_transform.hpp>
//#include <glm/gtc/type_ptr.hpp>
using namespace std;

/*
class CudaMeshRenderer {
public:
    GLuint vbo, ebo, vao;
    cudaGraphicsResource* cuda_vbo_resource;

    GLuint shaderProgram;

    int num_triangles;
    glm::mat4 model, view, projection;

    CudaMeshRenderer() : vbo(0), ebo(0), vao(0), cuda_vbo_resource(nullptr) {}

    void init(const trimesh& m, float aspect_ratio) {
        num_triangles = m.num_triangles;

        // === Upload vertex buffer ===
        glGenBuffers(1, &vbo);
        glBindBuffer(GL_ARRAY_BUFFER, vbo);
        glBufferData(GL_ARRAY_BUFFER, m.num_vertices * sizeof(float3), m.vertices, GL_DYNAMIC_DRAW);
        glBindBuffer(GL_ARRAY_BUFFER, 0);

        // === Upload index buffer ===
        glGenBuffers(1, &ebo);
        glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, ebo);
        glBufferData(GL_ELEMENT_ARRAY_BUFFER, m.num_triangles * sizeof(int3), m.triangles, GL_STATIC_DRAW);
        glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, 0);

        // === Register with CUDA ===
        cudaGraphicsGLRegisterBuffer(&cuda_vbo_resource, vbo, cudaGraphicsMapFlagsWriteDiscard);

        // === Setup VAO ===
        glGenVertexArrays(1, &vao);
        glBindVertexArray(vao);
        glBindBuffer(GL_ARRAY_BUFFER, vbo);
        glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, sizeof(float3), (void*)0);
        glEnableVertexAttribArray(0);
        glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, ebo);
        glBindVertexArray(0);

        // === Setup shader ===
        shaderProgram = compileShaderProgram(); // Define this

        // === Setup camera matrices ===
        model = glm::mat4(1.0f);
        view = glm::lookAt(glm::vec3(0, 0, 5), glm::vec3(0, 0, 0), glm::vec3(0, 1, 0));
        projection = glm::perspective(glm::radians(45.0f), aspect_ratio, 0.1f, 100.0f);
    }

    float3* mapCudaPointer() {
        float3* dptr = nullptr;
        size_t num_bytes = 0;
        cudaGraphicsMapResources(1, &cuda_vbo_resource, 0);
        cudaGraphicsResourceGetMappedPointer((void**)&dptr, &num_bytes, cuda_vbo_resource);
        return dptr;
    }

    void unmapCudaPointer() {
        cudaGraphicsUnmapResources(1, &cuda_vbo_resource, 0);
    }

    void render(float rotation_angle = 0.0f) {
        glUseProgram(shaderProgram);

        // update model rotation
        model = glm::rotate(glm::mat4(1.0f), rotation_angle, glm::vec3(0, 1, 0));

        // send uniforms
        glUniformMatrix4fv(glGetUniformLocation(shaderProgram, "model"), 1, GL_FALSE, glm::value_ptr(model));
        glUniformMatrix4fv(glGetUniformLocation(shaderProgram, "view"), 1, GL_FALSE, glm::value_ptr(view));
        glUniformMatrix4fv(glGetUniformLocation(shaderProgram, "projection"), 1, GL_FALSE, glm::value_ptr(projection));

        glBindVertexArray(vao);
        glDrawElements(GL_TRIANGLES, num_triangles * 3, GL_UNSIGNED_INT, 0);
        glBindVertexArray(0);
    }

    ~CudaMeshRenderer() {
        if (cuda_vbo_resource) cudaGraphicsUnregisterResource(cuda_vbo_resource);
        glDeleteBuffers(1, &vbo);
        glDeleteBuffers(1, &ebo);
        glDeleteVertexArrays(1, &vao);
        glDeleteProgram(shaderProgram);
    }

private:
    GLuint compileShaderProgram() {
        // Replace with actual shader compilation
        // Load vertex and fragment shader source, compile, link
        // Must define `model`, `view`, `projection` as uniforms in vertex shader
        // Use simple hardcoded shader if needed
        // Return compiled program ID
        return simpleShaderProgram(); // placeholder
    }

    GLuint simpleShaderProgram(); // you can fill this or I can give you the code
};
*/

std::vector<std::string> split(std::string s, std::string delimiter) {
    size_t pos_start = 0, pos_end, delim_len = delimiter.length();
    std::string token;
    std::vector<std::string> res;

    while ((pos_end = s.find(delimiter, pos_start)) != std::string::npos) {
        token = s.substr (pos_start, pos_end - pos_start);
        pos_start = pos_end + delim_len;
        res.push_back (token);
    }

    res.push_back (s.substr (pos_start));
    return res;
}
void read_basic_ply(string filename, float3*& vertices, int3*& triangles, int& num_vertices, int& num_triangles)
{
    ifstream ifs;
    ifs.open(filename);
    std::string line;
    map<string, int> elements;
    while(std::getline(ifs, line))
    {
        if(line == "ply")
        {
            continue;
        }
        else if(line == "end_header")
        {
            break;
        }
        else
        {
            if(line.find("element", 0) == 0)
            {
                vector<string> bits = split(line, " ");
                if(bits.size() == 3)
                {
                    elements[bits[1]] = stoi(bits[2]);
                }
            }
        }
    };
    num_vertices = elements["vertex"];
    num_triangles = elements["face"];
    vertices = (float3*)malloc(sizeof(float3)*num_vertices);
    triangles = (int3*)malloc(sizeof(int3)*num_triangles);
    int v_idx = 0;
    int t_idx = 0;
    while(std::getline(ifs, line))
    {
        vector<string> bits = split(line, " ");
        if(bits.size() == 3)
        {   
            vertices[v_idx] = make_float3(stof(bits[0]), stof(bits[1]), stof(bits[2]));
            v_idx += 1;
        }
        else if(bits.size() == 4)
        {
            if(bits[0] == "3")
            {
                triangles[t_idx] = make_int3(stoi(bits[1]), stoi(bits[2]), stoi(bits[3]));
                t_idx += 1;
            }
        }
        else
        {
            continue;
        }
    }
    ifs.close();
}   

int main()
{
    float3* host_vertices;
    int3* host_triangles;
    int num_vertices;
    int num_triangles;
    read_basic_ply("test.ply",host_vertices, host_triangles, num_vertices, num_triangles);
    trimesh host_mesh;
    vector<vector<int>> boundary_loops_vertices;
    vector<vector<int>> boundary_loops_edge_indices;
    trimesh_prepare(host_vertices, host_triangles, num_vertices, num_triangles, host_mesh, boundary_loops_vertices, boundary_loops_edge_indices, false);
    trimesh mesh;
    trimesh_host_to_device(host_mesh, mesh);
    int threads_per_block = 128;
    int edge_blocks = (mesh.num_edges + threads_per_block - 1) / threads_per_block;
    int triangle_blocks = (mesh.num_triangles + threads_per_block - 1) / threads_per_block;
    float* edge_values;
    cudaMalloc(&edge_values, sizeof(float) * mesh.num_edges);
    zero<<<edge_blocks, threads_per_block>>>(edge_values, mesh.num_edges);
    cudaDeviceSynchronize();
    get_cotan_weights<<<edge_blocks, threads_per_block>>>(edge_values, mesh);
    cudaDeviceSynchronize();
    float3* tri_normals;
    cudaMalloc(&tri_normals, sizeof(float3) * mesh.num_triangles);
    compute_triangle_normals<<<triangle_blocks, threads_per_block>>>(mesh.vertices, mesh.triangles, tri_normals, mesh.num_triangles);
    cudaDeviceSynchronize();

    //float* h_edge_values = copy_to_host(edge_values, mesh.num_edges);
    //for(int e_idx = 0; e_idx < mesh.num_edges; e_idx++)
    //{
    //    if(h_edge_values[e_idx] == 0.0)
    //    {
    //        cout << host_mesh.edge_from[e_idx] << " " << host_mesh.edge_to[e_idx] << " " <<  h_edge_values[e_idx] << endl;
    //    }
    //    
    //}
    return 0;
};