#pragma once
#include <cmath>
#include <limits>
#include <algorithm>
using namespace std;
static constexpr float inf = std::numeric_limits<float>::infinity();
static constexpr float quiet_nan = std::numeric_limits<float>::quiet_NaN();



struct float2
{
    float x;
    float y;

    float2()
    {
        x = 0.0f;
        y = 0.0f;
    };

    float2(float x_, float y_)
    {
        x = x_;
        y = y_;
    };
};
struct float3
{
    float x;
    float y;
    float z;

    float3()
    {
        x = 0.0f;
        y = 0.0f;
        z = 0.0f;
    };

    float3(float x_, float y_, float z_)
    {
        x = x_;
        y = y_;
        z = z_;
    };
};
struct int2
{
    float x;
    float y;

    int2()
    {
        x = 0;
        y = 0;
    };

    int2(int x_, int y_)
    {
        x = x_;
        y = y_;
    };
};
struct int3
{
    float x;
    float y;
    float z;

    int3()
    {
        x = 0;
        y = 0;
        z = 0;
    };

    int3(int x_, int y_, int z_)
    {
        x = x_;
        y = y_;
        z = z_;
    };
};
struct box2
{
    float2 pmin = {inf,inf};
    float2 pmax = {-inf,-inf};

    box2 & operator |= (float2 const & p)
    {
        pmin.x = min(pmin.x, p.x);
        pmin.y = min(pmin.y, p.y);
        pmax.x = max(pmax.x, p.x);
        pmax.y = max(pmax.y, p.y);
        return *this;
    };
};

static constexpr float2 zero2 = {0.0f,0.0f};
static constexpr float3 zero3 = {0.0f,0.0f,0.0f};
static constexpr float2 nan2 = {quiet_nan,quiet_nan};
static constexpr float3 nan3 = {quiet_nan,quiet_nan,quiet_nan};


float3 make_float3(float x_, float y_, float z_)
{
    return float3(x_,y_,z_);
};

template <typename Iterator>
box2 compute_bbox(Iterator begin, Iterator end)
{
    box2 result;
    for (auto it = begin; it != end; ++it)
        result |= *it;
    return result;
}


inline int pmod(int i, int n)
{
    return (i % n + n) % n;
};
inline float fpmod(float x, float l)
{
    return fmodf(fmodf(x,l) + l, l);
};
inline float3 operator+(const float3& a, const float3& b) {
    return float3(a.x + b.x, a.y + b.y, a.z + b.z);
};
inline float3& operator+=(float3& a, const float3& b) {
    a.x += b.x;
    a.y += b.y;
    a.z += b.z;
    return a;
};
inline float2 operator+(const float2& a, const float2& b) {
    return float2(a.x + b.x, a.y + b.y);
};
inline float3 operator-(const float3& a, const float3& b) {
    return float3(a.x - b.x, a.y - b.y, a.z - b.z);
};
inline float3& operator-=(float3& a, const float3& b) {
    a.x -= b.x;
    a.y -= b.y;
    a.z -= b.z;
    return a;
};
inline float3 operator*(const float3& a, const float s) {
    return float3(a.x * s, a.y * s, a.z * s);
};
inline float3& operator*=(float3& a, const float s) {
    a.x *= s;
    a.y *= s;
    a.z *= s;
    return a;
};
inline float3 operator*(const float s, const float3& a) {
    return a * s;
};
inline float3 operator/(const float3& a, const float s) {
    float inv = 1.0f / s;
    return a * inv;
};
inline float3& operator/=(float3& a, const float s) {
    float inv = 1.0f / s;
    a *= inv;
    return a;
};
inline bool operator==(const float3& a, const float3& b)
{
    return (a.x == b.x) && (a.y == b.y) && (a.z == b.z);
}
inline float3 cross(const float3& u, const float3& v)
{
    return float3(u.y*v.z - u.z*v.y,
        u.z*v.x - u.x*v.z,
        u.x*v.y - u.y*v.x);
};
inline float dot(const float3& u, const float3& v)
{
    return u.x*v.x + u.y*v.y + u.z*u.z;
};
inline float3 rodrigues(const float3& v, const float3& n, const float theta)
{
    float3 vec1 = cross(n,v);
    float3 vec2 = cross(n, vec1);
    float coef1 = sin(theta);
    float coef2 = 1- cos(theta);
    return v + coef1*vec1 + coef2*vec2;
};
inline float length(const float3& a) {
    return sqrtf(dot(a, a));
};
inline float3 normalize(const float3& a) {
    return a / length(a);
};
inline float cotangent(float3 u, float3 v) {
    float dot_uv = dot(u, v);
    float3 cross_uv = cross(u, v);
    float cross_norm = length(cross_uv);
    return dot_uv / cross_norm;  // avoid divide-by-zero
}
inline float2 midpoint(const float2& a, const float2& b)
{
    return float2((a.x+b.x)/2.0f, (a.y+b.y)/2.0f);
};
inline float3 midpoint(const float3& a, const float3& b)
{
    return float3((a.x+b.x)/2.0f, (a.y+b.y)/2.0f, (a.z+b.z)/2.0f);
};


