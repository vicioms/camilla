#pragma once
#include <cmath>
#include <limits>
#include <algorithm>

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

const int2 zero_int2 = {0,0};
const int3 zero_int3 = {0,0,0};
const float2 zero_float2 = {0.0f,0.0f};
const float3 zero_float3 = {0.0f,0.0f,0.0f};

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

inline int pmod(int i, int n)
{
    return (i % n + n) % n;
};
inline float wrap(float x, float l)
{
    return x - l*rintf(x/l);
};
inline float wrap(float x, float x_min, float x_max)
{
    return x_min + wrap(x - x_min, x_max - x_min);
};
inline float2 wrap(float2 v, float2 box_min, float2 box_max)
{
    return make_float2(
        wrap(v.x, box_min.x, box_max.x),
        wrap(v.y, box_min.y, box_max.y)
    );
};
inline float3 wrap(float3 x, float3 box_min, float3 box_max)
{
    return make_float3(
        wrap(x.x, box_min.x, box_max.x),
        wrap(x.y, box_min.y, box_max.y),
        wrap(x.z, box_min.z, box_max.z)
    );
};
inline float2 wrap_diff(float2 a, float2 b, float2 box_size)
{
    return make_float2(wrap(a.x-b.x, box_size.x), wrap(a.y-b.y, box_size.y));
};
inline float3 wrap_diff(float3 a, float3 b, float3 box_size)
{
    return make_float3(wrap(a.x-b.x, box_size.x), wrap(a.y-b.y, box_size.y), wrap(a.z-b.z, box_size.z));
};



inline float2 operator+(const float2& a, const float2& b)
{
    return float2(a.x + b.x, a.y + b.y);
};
inline float3 operator+(const float3& a, const float3& b)
{
    return float3(a.x + b.x, a.y + b.y, a.z + b.z);
};
inline float2 operator+(const float2& a, float b)
{
    return make_float2(a.x + b, a.y + b);
};
inline float2 operator+(float a, const float2& b)
{
    return make_float2(a + b.x, a + b.y);
};
inline float3 operator+(const float3& a, float b)
{
    return make_float3(a.x + b, a.y + b, a.z + b);
};
inline float3 operator+(float a, const float3& b)
{
    return make_float3(a + b.x, a + b.y, a + b.z);
};
inline float2& operator+=(float2& a, const float2& b) {
    a.x += b.x;
    a.y += b.y;
    return a;
};
inline float2& operator+=(float2& a, float b) {
    a.x += b;
    a.y += b;
    return a;
};
inline float3& operator+=(float3& a, const float3& b) {
    a.x += b.x;
    a.y += b.y;
    a.z += b.z;
    return a;
};
inline float3& operator+=(float3& a, float b) {
    a.x += b;
    a.y += b;
    a.z += b;
    return a;
};


inline float2 operator-(const float2& a, const float2& b)
{
    return float2(a.x - b.x, a.y - b.y);
};
inline float3 operator-(const float3& a, const float3& b)
{
    return float3(a.x - b.x, a.y - b.y, a.z - b.z);
};
inline float2 operator-(const float2& a, float b)
{
    return make_float2(a.x - b, a.y - b);
};
inline float2 operator-(float a, const float2& b)
{
    return make_float2(a - b.x, a - b.y);
};
inline float3 operator-(const float3& a, float b)
{
    return make_float3(a.x - b, a.y - b, a.z - b);
};
inline float3 operator-(float a, const float3& b)
{
    return make_float3(a - b.x, a - b.y, a - b.z);
};
inline float2& operator-=(float2& a, const float2& b) {
    a.x -= b.x;
    a.y -= b.y;
    return a;
};
inline float2& operator-=(float2& a, float b) {
    a.x -= b;
    a.y -= b;
    return a;
};
inline float3& operator-=(float3& a, const float3& b) {
    a.x -= b.x;
    a.y -= b.y;
    a.z -= b.z;
    return a;
};
inline float3& operator-=(float3& a, float b) {
    a.x -= b;
    a.y -= b;
    a.z -= b;
    return a;
};


inline float2 operator*(const float2& a, float b)
{
    return make_float2(a.x * b, a.y * b);
};
inline float2 operator*(float a, const float2& b)
{
    return make_float2(a * b.x, a * b.y);
};
inline float3 operator*(const float3& a, float b)
{
    return make_float3(a.x * b, a.y * b, a.z * b);
};
inline float3 operator*(float a, const float3& b)
{
    return make_float3(a * b.x, a * b.y, a * b.z);
};
inline float2& operator*=(float2& a, float b) {
    a.x *= b;
    a.y *= b;
    return a;
};
inline float3& operator*=(float3& a, float b) {
    a.x *= b;
    a.y *= b;
    a.z *= b;
    return a;
};


inline float2 operator/(const float2& a, float b)
{
    float b_inv = 1.0f / b;
    return make_float2(a.x * b_inv, a.y * b_inv);   
};
inline float2 operator/(float a, const float2& b)
{
    return make_float2(a / b.x, a / b.y);
};
inline float3 operator/(const float3& a, float b)
{
    float b_inv = 1.0f / b;
    return make_float3(a.x * b_inv, a.y * b_inv, a.z * b_inv);
};
inline float3 operator/(float a, const float3& b)
{
    return make_float3(a / b.x, a / b.y, a / b.z);
};
inline float2& operator/=(float2& a, float b) {
    a.x /= b;
    a.y /= b;
    return a;
};
inline float3& operator/=(float3& a, float b) {
    a.x /= b;
    a.y /= b;
    a.z /= b;
    return a;
};

float dot(const float2& a, const float2& b)
{
    return a.x*b.x + a.y*b.y;
};
float dot(const float3& a, const float3& b)
{
    return a.x*b.x + a.y*b.y + a.z*b.z;
};
float length(const float2& v)
{
    return sqrtf(v.x*v.x + v.y*v.y);
};
float length(const float3& v)
{
    return sqrtf(v.x*v.x + v.y*v.y + v.z*v.z);
};
float2 normalize(const float2& v)
{
    float len = length(v);
    return v/len;
};
float3 normalize(const float3& v)
{
    float len = length(v);
    return v/len;
};
