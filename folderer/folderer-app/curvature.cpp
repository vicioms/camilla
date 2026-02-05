#include <vector>
#include <cmath>
#include <emscripten/emscripten.h>
#include <emscripten/bind.h>
#include <cstddef>
#include <cstdint>

void area_vector(const float* a, const float* b, const float* c, float* areaVec)
{
    float ab[3] = {b[0] - a[0], b[1] - a[1], b[2] - a[2]};
    float ac[3] = {c[0] - a[0], c[1] - a[1], c[2] - a[2]};
    areaVec[0] = 0.5f * (ab[1] * ac[2] - ab[2] * ac[1]);
    areaVec[1] = 0.5f * (ab[2] * ac[0] - ab[0] * ac[2]);
    areaVec[2] = 0.5f * (ab[0] * ac[1] - ab[1] * ac[0]);
}

float norm(const float* vec)
{
    return std::sqrt(vec[0] * vec[0] + vec[1] * vec[1] + vec[2] * vec[2]);
}
float dot(const float* a, const float* b)
{
    return a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
}

class TriangleMesh {
public:
  float* vertices;
  uint32_t* faces;
  size_t vertexCount;
  size_t faceCount;
  
  
  TriangleMesh(float* verts, uint32_t* tris, size_t vCount, size_t fCount)
    : vertices(verts), faces(tris), vertexCount(vCount), faceCount(fCount) {}

  void compute_curvatures(float* mean_curvature, float* pc1, float* pc2) const
  {
        std::vector<float> tri_cots(3*faceCount, 0.f);
        std::vector<float> tri_areas(faceCount, 0.f);
        for(size_t face_idx = 0; face_idx < faceCount; ++face_idx) {
            uint32_t v0_idx = faces[3 * face_idx + 0];
            uint32_t v1_idx = faces[3 * face_idx + 1];
            uint32_t v2_idx = faces[3 * face_idx + 2];
            float* v0 = &vertices[3 * v0_idx];
            float* v1 = &vertices[3 * v1_idx];
            float* v2 = &vertices[3 * v2_idx];
            float areaVec[3];
            area_vector(v0, v1, v2, areaVec);
            float area = norm(areaVec);
            tri_areas[face_idx] = area;

            float edge_01[3] = {v1[0] - v0[0], v1[1] - v0[1], v1[2] - v0[2]};
            float edge_02[3] = {v2[0] - v0[0], v2[1] - v0[1], v2[2] - v0[2]};
            float edge_12[3] = {v2[0] - v1[0], v2[1] - v1[1], v2[2] - v1[2]};
            float cot_0 = dot(edge_01, edge_02) / (2.0f*area);
            float cot_1 = -dot(edge_01, edge_12) / (2.0f*area);
            float cot_2 = dot(edge_02, edge_12) / (2.0f*area);
            tri_cots[3*face_idx + 0] = cot_0;
            tri_cots[3*face_idx + 1] = cot_1;
            tri_cots[3*face_idx + 2] = cot_2;
        }
  }


};