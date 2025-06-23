#pragma once
#include <cmath>
#include <limits>
#include <algorithm>
using namespace std;
static constexpr float inf = std::numeric_limits<float>::infinity();
static constexpr float quiet_nan = std::numeric_limits<float>::quiet_NaN();


inline int pmod(int i, int n)
{
    return (i % n + n) % n;
};
inline float fpmod(float x, float l)
{
    return fmodf(fmodf(x,l) + l, l);
};

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
    int x;
    int y;

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

    bool operator==(const int2& other) const {
        return x == other.x && y == other.y;
    }
};
struct int3
{
    int x;
    int y;
    int z;

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

    bool operator==(const int3& other) const {
        return x == other.x && y == other.y && z == other.z;
    }
};


const float2 zero2 = {0.0f,0.0f};
const float3 zero3 = {0.0f,0.0f,0.0f};
const float2 nan2 = {quiet_nan,quiet_nan};
const float3 nan3 = {quiet_nan,quiet_nan,quiet_nan};


int2 make_int2(int x_, int y_)
{
    return int2(x_,y_);
};
int3 make_int3(int x_, int y_, int z_)
{
    return int3(x_,y_,z_);
};
float2 make_float2(float x_, float y_)
{
    return float2(x_,y_);
};
float3 make_float3(float x_, float y_, float z_)
{
    return float3(x_,y_,z_);
};


inline float2 operator+(const float2& a, const float2& b) {
    return float2(a.x - b.x, a.y - b.y);
};
inline float2 operator-(const float2& a, const float2& b) {
    return float2(a.x + b.x, a.y + b.y);
};
inline float2 midpoint(const float2& a, const float2& b)
{
    return float2((a.x+b.x)/2.0f, (a.y+b.y)/2.0f);
};
inline float dot(const float2& u, const float2& v)
{
    return u.x*v.x + u.y*v.y;
};
inline float cross(const float2& u, const float2& v)
{
    return u.x*v.y - u.y*v.x;
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
inline float3 midpoint(const float3& a, const float3& b)
{
    return float3((a.x+b.x)/2.0f, (a.y+b.y)/2.0f, (a.z+b.z)/2.0f);
};
inline bool inside_circumcircle(const float2& a, const float2& b, const float2& c, const float2& p)
{
    //assumes ccw orientation of a, b, c
    float a_1 = a.x - p.x;
    float a_2 = a.y - p.y;
    float a_3 = a_1*a_1 + a_2*a_2;
    float b_1 = b.x - p.x;
    float b_2 = b.y - p.y;
    float b_3 = b_1*b_1 + b_2*b_2;
    float c_1 = c.x - p.x;
    float c_2 = c.y - p.y;
    float c_3 = c_1*c_1 + c_2*c_2;

    float det = a_1*(b_2*c_3 - b_3*c_2) -a_2*(b_1*c_3 - c_1*b_3) + a_3*(b_1*c_2 - c_1*b_2);
    return det > 0;
};
inline bool is_right_of(const float2& x, const float2& source, const float2& target)
{
    return cross(x - source, target-source) > 0;
};
