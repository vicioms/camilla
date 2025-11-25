#pragma once
#include "helper_math.h"

struct mat2
{
    float xx;
    float xy;
    float yx;
    float yy;
};

struct smat2
{
    float xx;
    float xy;
    float yy;
};

inline __host__ __device__ mat2 make_mat2(float val)
{
    mat2 m;
    m.xx = val; m.xy = val;
    m.yx = val; m.yy = val;
    return m;
};
inline __host__ __device__ mat2 make_mat2(float m_xx, float m_xy, float m_yx, float m_yy)
{
    mat2 m;
    m.xx = m_xx; m.xy = m_xy;
    m.yx = m_yx; m.yy = m_yy;
    return m;
};

inline __host__ __device__ smat2 make_smat2(float m_xx, float m_xy, float m_yy)
{
    smat2 m;
    m.xx = m_xx; m.xy = m_xy;
    m.yy = m_yy;
    return m;
};
inline __host__ __device__ smat2 make_smat2(float val)
{
    smat2 m;
    m.xx = val; m.xy = val;
    m.yy = val;
    return m;
};

inline __host__ __device__ mat2 make_mat2(smat2 m)
{
    return make_mat2(m.xx, m.xy,
                     m.xy, m.yy);
};

////////////////////////////////////////////////////////////////////////////////
// negate
////////////////////////////////////////////////////////////////////////////////

inline __host__ __device__ mat2 operator-(mat2 m)
{
    return make_mat2(-m.xx, -m.xy, -m.yx, -m.yy);
};
inline __host__ __device__ smat2 operator-(smat2 m)
{
    return make_smat2(-m.xx, -m.xy, -m.yy);
};

////////////////////////////////////////////////////////////////////////////////
// addition
////////////////////////////////////////////////////////////////////////////////

inline __host__ __device__ mat2 operator+(mat2 a, mat2 b)
{
    return make_mat2(a.xx + b.xx, a.xy + b.xy, a.yx + b.yx, a.yy + b.yy);
};
inline __host__ __device__ void operator+=(mat2& a, mat2 b)
{
   a.xx += b.xx;
   a.xy += b.xy;
   a.yx += b.yx;
   a.yy += b.yy;
};
inline __host__ __device__ mat2 operator+(mat2 a, float b)
{
    return make_mat2(a.xx + b, a.xy + b, a.yx + b, a.yy + b);
};
inline __host__ __device__ mat2 operator+(float a, mat2 b)
{
    return make_mat2(a + b.xx, a + b.xy, a + b.yx, a + b.yy);
};
inline __host__ __device__ void operator+=(mat2& a, float b)
{
   a.xx += b;
   a.xy += b;
   a.yx += b;
   a.yy += b;
};
inline __host__ __device__ smat2 operator+(smat2 a, smat2 b)
{
    return make_smat2(a.xx + b.xx, a.xy + b.xy, a.yy + b.yy);
};
inline __host__ __device__ void operator+=(smat2& a, smat2 b)
{
   a.xx += b.xx;
   a.xy += b.xy;
   a.yy += b.yy;
};
inline __host__ __device__ smat2 operator+(smat2 a, float b)
{
    return make_smat2(a.xx + b, a.xy + b, a.yy + b);
};
inline __host__ __device__ smat2 operator+(float a, smat2 b)
{
    return make_smat2(a + b.xx, a + b.xy, a + b.yy);
};
inline __host__ __device__ void operator+=(smat2& a, float b)
{
   a.xx += b;
   a.xy += b;
   a.yy += b;
};



////////////////////////////////////////////////////////////////////////////////
// subtraction
////////////////////////////////////////////////////////////////////////////////

inline __host__ __device__ mat2 operator-(mat2 a, mat2 b)
{
    return make_mat2(a.xx - b.xx, a.xy - b.xy, a.yx - b.yx, a.yy - b.yy);
};
inline __host__ __device__ void operator-=(mat2& a, mat2 b)
{
   a.xx -= b.xx;
   a.xy -= b.xy;
   a.yx -= b.yx;
   a.yy -= b.yy;
};
inline __host__ __device__ mat2 operator-(mat2 a, float b)
{
    return make_mat2(a.xx - b, a.xy - b, a.yx - b, a.yy - b);
};
inline __host__ __device__ mat2 operator-(float a, mat2 b)
{
    return make_mat2(a - b.xx, a - b.xy, a - b.yx, a - b.yy);
};
inline __host__ __device__ void operator-=(mat2& a, float b)
{
   a.xx -= b;
   a.xy -= b;
   a.yx -= b;
   a.yy -= b;
};
inline __host__ __device__ smat2 operator-(smat2 a, smat2 b)
{
    return make_smat2(a.xx - b.xx, a.xy - b.xy, a.yy - b.yy);
};
inline __host__ __device__ void operator-=(smat2& a, smat2 b)
{
   a.xx -= b.xx;
   a.xy -= b.xy;
   a.yy -= b.yy;
};
inline __host__ __device__ smat2 operator-(smat2 a, float b)
{
    return make_smat2(a.xx - b, a.xy - b, a.yy - b);
};
inline __host__ __device__ smat2 operator-(float a, smat2 b)
{
    return make_smat2(a - b.xx, a - b.xy, a - b.yy);
};
inline __host__ __device__ void operator-=(smat2& a, float b)
{
   a.xx -= b;
   a.xy -= b;
   a.yy -= b;
};


////////////////////////////////////////////////////////////////////////////////
// multiply
////////////////////////////////////////////////////////////////////////////////

inline __host__ __device__ mat2 operator*(mat2 a, float b)
{
    return make_mat2(a.xx * b, a.xy * b,
                     a.yx * b, a.yy * b);
};
inline __host__ __device__ mat2 operator*(float a, mat2 b)
{
    return make_mat2(a * b.xx, a * b.xy,
                     a * b.yx, a * b.yy);
};
inline __host__ __device__ void operator*=(mat2& a, float b)
{
   a.xx *= b;
   a.xy *= b;
   a.yx *= b;
   a.yy *= b;
};
inline __host__ __device__ smat2 operator*(smat2 a, float b)
{
    return make_smat2(a.xx * b, a.xy * b,
                    a.yy * b);
};
inline __host__ __device__ smat2 operator*(float a, smat2 b)
{
    return make_smat2(a * b.xx, a * b.xy,
                    a * b.yy);
};
inline __host__ __device__ void operator*=(smat2& a, float b)
{
   a.xx *= b;
   a.xy *= b;
   a.yy *= b;
};

////////////////////////////////////////////////////////////////////////////////
// misc
////////////////////////////////////////////////////////////////////////////////
inline __host__ __device__ mat2 matmul(mat2 a, mat2 b)
{
    return make_mat2(a.xx * b.xx + a.xy * b.yx,
                     a.xx * b.xy + a.xy * b.yy,
                     a.yx * b.xx + a.yy * b.yx,
                     a.yx * b.xy + a.yy * b.yy);
};
inline __host__ __device__ float2 matmul(mat2 a, float2 b)
{
    return make_float2(a.xx * b.x + a.xy * b.y,
                     a.yx * b.x + a.yy * b.y);
};
inline __host__ __device__ mat2 transpose(mat2 m)
{
    return make_mat2(m.xx, m.yx,
                      m.xy, m.yy);
};
inline __host__ __device__ float det(mat2 m)
{
    return m.xx * m.yy - m.xy * m.yx;
};
inline __host__ __device__ mat2 inverse(mat2 m)
{
    float determinant = det(m);
    if (determinant == 0) return make_mat2(0.0f);
    float inv_det = 1.0f / determinant;
    return make_mat2( m.yy * inv_det, -m.yx * inv_det, -m.xy * inv_det, m.xx * inv_det);
};
inline __host__ __device__ smat2 square(smat2 a)
{
    return make_smat2(a.xx * a.xx + a.xy * a.xy,
                     a.xx * a.xy + a.xy * a.yy,
                     a.xy * a.xy + a.yy * a.yy);
};
inline __host__ __device__ float2 matmul(smat2 a, float2 b)
{
    return make_float2(a.xx * b.x + a.xy * b.y,
                     a.xy * b.x + a.yy * b.y);
};
inline __host__ __device__ float det(smat2 m)
{
    return m.xx * m.yy - m.xy * m.xy;
};
inline __host__ __device__ smat2 inverse(smat2 m)
{
    float determinant = det(m);
    if (determinant == 0) return make_smat2(0.0f);
    float inv_det = 1.0f / determinant;
    return make_smat2( m.yy * inv_det, -m.xy * inv_det, m.xx * inv_det);
};
inline __host__ __device__ smat2 outer(float2 v)
{
    return make_smat2(v.x * v.x, v.x * v.y, v.y * v.y);
};
