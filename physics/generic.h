#pragma once
#include <cmath>
#include <limits>
#include <algorithm>

static constexpr float inf = std::numeric_limits<float>::infinity();
static constexpr float quiet_nan = std::numeric_limits<float>::quiet_NaN();


inline int pmod(int i, int n)
{
    return (i % n + n) % n;
};

inline float wrap(float x, float l)
{
    return x - l*rintf(x/l);
}
inline float wrap(float x, float x_min, float x_max)
{
    return x_min + wrap(x - x_min, x_max - x_min);
};


inline int pack(int r, int c, int dim)
{
    return r*dim+c;
};
inline void unpack(int idx, int dim, int& r, int& c)
{
    r = idx / dim;
    c = pmod(idx, dim);
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


// float2 operators
inline float2 operator+(const float2& a, const float2& b) {
    return float2(a.x + b.x, a.y + b.y);
};
inline float2 operator-(const float2& a, const float2& b) {
    return float2(a.x - b.x, a.y - b.y);
};
inline float2 operator*(const float2& a, const float s) {
    return float2(a.x * s, a.y * s);
};
inline float2 operator*(const float s, const float2& a) {
    return a * s;
};
inline float2 operator/(const float2& a, const float s) {
    float inv = 1.0f / s;
    return a * inv;
};
inline bool operator==(const float2& a, const float2& b)
{
    return (a.x == b.x) && (a.y == b.y);
};
inline bool operator!=(const float2& a, const float2& b)
{
    return !(a == b);
};
inline float2& operator+=(float2& a, const float2& b) {
    a.x += b.x;
    a.y += b.y;
    return a;
};
inline float2& operator-=(float2& a, const float2& b) {
    a.x -= b.x;
    a.y -= b.y;
    return a;
};
inline float2& operator*=(float2& a, const float s) {
    a.x *= s;
    a.y *= s;
    return a;
};
inline float2& operator/=(float2& a, const float s) {
    float inv = 1.0f / s;
    a *= inv;
    return a;
};


// float3 operators
inline float3 operator+(const float3& a, const float3& b) {
    return float3(a.x + b.x, a.y + b.y, a.z + b.z);
};
inline float3 operator-(const float3& a, const float3& b) {
    return float3(a.x - b.x, a.y - b.y, a.z - b.z);
};
inline float3 operator*(const float3& a, const float s) {
    return float3(a.x * s, a.y * s, a.z * s);
};
inline float3 operator*(const float s, const float3& a) {
    return a * s;
};
inline float3 operator/(const float3& a, const float s) {
    float inv = 1.0f / s;
    return a * inv;
};

inline float3& operator+=(float3& a, const float3& b) {
    a.x += b.x;
    a.y += b.y;
    a.z += b.z;
    return a;
};
inline float3& operator-=(float3& a, const float3& b) {
    a.x -= b.x;
    a.y -= b.y;
    a.z -= b.z;
    return a;
};
inline float3& operator*=(float3& a, const float s) {
    a.x *= s;
    a.y *= s;
    a.z *= s;
    return a;
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


// useful float2 funcs

inline float2 midpoint(const float2& a, const float2& b)
{
    return float2((a.x+b.x)/2.0f, (a.y+b.y)/2.0f);
};
inline float dot(const float2& u, const float2& v)
{
    return u.x*v.x + u.y*v.y;
};
inline float length(const float2& a) {
    return sqrtf(dot(a, a));
};
inline float cross(const float2& u, const float2& v)
{
    return u.x*v.y - u.y*v.x;
};
inline float polygon_area(float2* p, int n)
{
    float s = 0;
    for(int i = 0; i < n; i++)
    {
        int i_next = pmod(i+1,n);
        s += p[i].x*p[i_next].y - p[i].y * p[i_next].x;
    };
    return 0.5*s;
};
inline void polygon_area_gradients(float2* p, float2* grad, int n, float prefactor, bool accumulate)
{
    if(accumulate == false)
    {
        for(int i = 0; i < n; i++)
        {
            grad[i] = zero2;
        };
    };
    for(int i = 0; i < n; i++)
    {
        //as in area gradients,
        //a vertex takes a grad equal to the 90 degs rotation
        //of the next vertex, negated (or a -90 degs rotation)
        //indeed the 
        int i_next = pmod(i+1,n);
        int i_prev = pmod(i-1,n);
        grad[i].x += 0.5f * prefactor*(p[i_next].y - p[i_prev].y); 
        grad[i].y += -0.5f * prefactor*(p[i_next].x - p[i_prev].x);
    };
};
inline float quad_area_signed(const float2& v1, const float2& v2, const float2& v3, const float2& v4)
{
    float s = v1.x*v2.y - v1.y*v2.x
            + v2.x*v3.y - v2.y*v3.x
            + v3.x*v4.y - v3.y*v4.x
            + v4.x*v1.y - v4.y*v1.x;
    return 0.5f * s;
};
inline float tri_area_signed(const float2& v1, const float2& v2, const float2& v3)
{
    float s = v1.x*v2.y - v1.y*v2.x
            + v2.x*v3.y - v2.y*v3.x
            + v3.x*v1.y - v3.y*v1.x;
    return 0.5f * s;
};
inline void quad_area_gradients(const float2& v1, const float2& v2, const float2& v3, const float2& v4, float2& g1, float2& g2, float2& g3, float2& g4, float prefactor, bool accumulate)
{
    if(accumulate == false)
    {
        g1 = zero2;
        g2 = zero2;
        g3 = zero2;
        g4 = zero2;
    };
    g1.x += 0.5f * prefactor*(v2.y - v4.y); 
    g1.y += -0.5f * prefactor*(v2.x - v4.x);
    g2.x += 0.5f * prefactor*(v3.y - v1.y); 
    g2.y += -0.5f * prefactor*(v3.x - v1.x);
    g3.x += 0.5f * prefactor*(v4.y - v2.y); 
    g3.y += -0.5f * prefactor*(v4.x - v2.x);
    g4.x += 0.5f * prefactor*(v1.y - v3.y); 
    g4.y += -0.5f * prefactor*(v1.x - v3.x);
};
inline float2 normalize(const float2& a) {
    return a / length(a);
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
inline float2 apply_unit_vector_grad(const float2& r, const float2& v)
{
    float r_length = length(r);
    float2 r_hat = r/r_length;
    return (v - dot(v, r_hat)*r_hat)/r_length;
};
// useful float3 funcs
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


//generic functions
void fill(float* x, int n, float v)
{
    for(int i = 0; i < n; i++)
    {
        x[i] = v;
    };
};
void fill(float2* x, int n, const float2 v)
{
    for(int i = 0; i < n; i++)
    {
        x[i] = v;
    };
};
void fill(float3* x, int n, const float3 v)
{
    for(int i = 0; i < n; i++)
    {
        x[i] = v;
    };
};
void fill(int* x, int n, int v)
{
    for(int i = 0; i < n; i++)
    {
        x[i] = v;
    };
};
void fill(int2* x, int n, const int2 v)
{
    for(int i = 0; i < n; i++)
    {
        x[i] = v;
    };
};  
void fill(int3* x, int n, const int3 v)
{
    for(int i = 0; i < n; i++)
    {
        x[i] = v;
    };
};


int pbc_diff(int a, int b, int n)
{
    int d = a - b;
    if(d > n/2)
        d -= n;
    else if(d < -n/2)
        d += n;
    return d;
};
int2 pbc_diff(int2 a, int2 b, int2 n)
{
    return int2(pbc_diff(a.x, b.x, n.x), pbc_diff(a.y, b.y, n.y));
};

inline void get_closest_points_segments(
    const float2 p1, const float2 p2,
    const float2 q1, const float2 q2,
    float& s, float& t, float2& r_vec)
{
    const float2 u = p2 - p1;
    const float2 v = q2 - q1;
    const float2 w0 = p1 - q1;

    const float a = dot(u,u);      // u·u
    const float b = dot(u,v);      // u·v
    const float c = dot(v,v);      // v·v
    const float d = dot(u,w0);     // u·w0
    const float e = dot(v,w0);     // v·w0
    const float D = a*c - b*b;     // denominator

    float sN, sD = D, tN, tD = D;

    if (D < 1e-20f) {              // almost parallel
        sN = 0.0f;  sD = 1.0f;
        tN = e;     tD = c;
    } else {
        sN = (b*e - c*d);
        tN = (a*e - b*d);
        // clamp sN to [0, sD]
        if (sN < 0.0f) { sN = 0.0f; tN = e; tD = c; }
        else if (sN > sD) { sN = sD; tN = e + b; tD = c; }
    }

    // clamp tN (and recompute sN if needed)
    if (tN < 0.0f) {
        tN = 0.0f;
        if (-d < 0.0f)      sN = 0.0f;
        else if (-d > a)    sN = sD;
        else { sN = -d; sD = a; }
    } else if (tN > tD) {
        tN = tD;
        if ((-d + b) < 0.0f)       sN = 0.0f;
        else if ((-d + b) > a)     sN = sD;
        else { sN = (-d + b); sD = a; }
    }

    s = (fabsf(sD) < 1e-20f) ? 0.0f : (sN / sD);
    t = (fabsf(tD) < 1e-20f) ? 0.0f : (tN / tD);

    const float2 P = p1 + s * u;
    const float2 Q = q1 + t * v;
    r_vec = P - Q;
};