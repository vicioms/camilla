#pragma once
#include <cstdlib>
#include <new>
#include <stdexcept>
#include <cmath>
#include <limits>
#include <algorithm>

// Default: float
// Compile with -DUSE_FLOAT to switch everything to float.
#ifdef USE_FLOAT
using real = float;
#else
using real = double;
#endif

inline constexpr real R(double x)
{
    return static_cast<real>(x);
}

// -----------------------------------------------------------------------------
// basic vector / int structs
// -----------------------------------------------------------------------------

struct vec2
{
    real x;
    real y;

    vec2()
    {
        x = R(0.0);
        y = R(0.0);
    }

    vec2(real x_, real y_)
    {
        x = x_;
        y = y_;
    }
};

struct vec3
{
    real x;
    real y;
    real z;

    vec3()
    {
        x = R(0.0);
        y = R(0.0);
        z = R(0.0);
    }

    vec3(real x_, real y_, real z_)
    {
        x = x_;
        y = y_;
        z = z_;
    }
};

struct int2
{
    int x;
    int y;

    int2()
    {
        x = 0;
        y = 0;
    }

    int2(int x_, int y_)
    {
        x = x_;
        y = y_;
    }

    bool operator==(const int2& other) const
    {
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
    }

    int3(int x_, int y_, int z_)
    {
        x = x_;
        y = y_;
        z = z_;
    }

    bool operator==(const int3& other) const
    {
        return x == other.x && y == other.y && z == other.z;
    }
};

struct mat2
{
    real m00, m01;
    real m10, m11;

    mat2()
    {
        m00 = R(0.0);
        m01 = R(0.0);
        m10 = R(0.0);
        m11 = R(0.0);
    }

    mat2(real m00_, real m01_, real m10_, real m11_)
    {
        m00 = m00_;
        m01 = m01_;
        m10 = m10_;
        m11 = m11_;
    }
};

// -----------------------------------------------------------------------------
// constants and constructors
// -----------------------------------------------------------------------------

inline const int2 zero_int2 = {0, 0};
inline const int3 zero_int3 = {0, 0, 0};
inline const vec2 zero_vec2 = {R(0.0), R(0.0)};
inline const vec3 zero_vec3 = {R(0.0), R(0.0), R(0.0)};
inline const mat2 zero_mat2 = {R(0.0), R(0.0), R(0.0), R(0.0)};

inline int2 make_int2(int x_, int y_)
{
    return int2(x_, y_);
}

inline int3 make_int3(int x_, int y_, int z_)
{
    return int3(x_, y_, z_);
}

inline vec2 make_vec2(real x_, real y_)
{
    return vec2(x_, y_);
}

inline vec3 make_vec3(real x_, real y_, real z_)
{
    return vec3(x_, y_, z_);
}

inline mat2 make_mat2(real m00_, real m01_, real m10_, real m11_)
{
    return mat2(m00_, m01_, m10_, m11_);
}

// -----------------------------------------------------------------------------
// scalar helpers
// -----------------------------------------------------------------------------

inline int pmod(int i, int n)
{
    return (i % n + n) % n;
}


// ------------------------------------------------------------
// OLD FUNCTIONS: centered / minimum-image wrapping
// Maps to roughly [-L/2, L/2].
// Keep using these for pair differences.
// ------------------------------------------------------------

inline real wrap(real x, real l)
{
    return x - l * std::rint(x / l);
}

inline real wrap(real x, real x_min, real x_max)
{
    return x_min + wrap(x - x_min, x_max - x_min);
}

inline vec2 wrap(vec2 v, vec2 box_min, vec2 box_max)
{
    return make_vec2(
        wrap(v.x, box_min.x, box_max.x),
        wrap(v.y, box_min.y, box_max.y)
    );
}

inline vec3 wrap(vec3 x, vec3 box_min, vec3 box_max)
{
    return make_vec3(
        wrap(x.x, box_min.x, box_max.x),
        wrap(x.y, box_min.y, box_max.y),
        wrap(x.z, box_min.z, box_max.z)
    );
}

inline vec2 wrap_diff(vec2 a, vec2 b, vec2 box_size)
{
    return make_vec2(
        wrap(a.x - b.x, box_size.x),
        wrap(a.y - b.y, box_size.y)
    );
}

inline vec3 wrap_diff(vec3 a, vec3 b, vec3 box_size)
{
    return make_vec3(
        wrap(a.x - b.x, box_size.x),
        wrap(a.y - b.y, box_size.y),
        wrap(a.z - b.z, box_size.z)
    );
}


// ------------------------------------------------------------
// NEW FUNCTIONS: absolute box wrapping
// Maps positions to [x_min, x_max).
// Use these after position updates.
// ------------------------------------------------------------

inline real wrap_box(real x, real x_min, real x_max)
{
    real L = x_max - x_min;

    real y = std::fmod(x - x_min, L);

    if(y < R(0.0))
        y += L;

    return x_min + y;
}

inline vec2 wrap_box(vec2 v, vec2 box_min, vec2 box_max)
{
    return make_vec2(
        wrap_box(v.x, box_min.x, box_max.x),
        wrap_box(v.y, box_min.y, box_max.y)
    );
}

inline vec3 wrap_box(vec3 v, vec3 box_min, vec3 box_max)
{
    return make_vec3(
        wrap_box(v.x, box_min.x, box_max.x),
        wrap_box(v.y, box_min.y, box_max.y),
        wrap_box(v.z, box_min.z, box_max.z)
    );
}

// -----------------------------------------------------------------------------
// vec2 / vec3 operators
// -----------------------------------------------------------------------------

inline vec2 operator-(const vec2& v)
{
    return make_vec2(-v.x, -v.y);
}

inline vec3 operator-(const vec3& v)
{
    return make_vec3(-v.x, -v.y, -v.z);
}

inline vec2 operator+(const vec2& a, const vec2& b)
{
    return vec2(a.x + b.x, a.y + b.y);
}

inline vec3 operator+(const vec3& a, const vec3& b)
{
    return vec3(a.x + b.x, a.y + b.y, a.z + b.z);
}

inline vec2 operator+(const vec2& a, real b)
{
    return make_vec2(a.x + b, a.y + b);
}

inline vec2 operator+(real a, const vec2& b)
{
    return make_vec2(a + b.x, a + b.y);
}

inline vec3 operator+(const vec3& a, real b)
{
    return make_vec3(a.x + b, a.y + b, a.z + b);
}

inline vec3 operator+(real a, const vec3& b)
{
    return make_vec3(a + b.x, a + b.y, a + b.z);
}

inline vec2& operator+=(vec2& a, const vec2& b)
{
    a.x += b.x;
    a.y += b.y;
    return a;
}

inline vec2& operator+=(vec2& a, real b)
{
    a.x += b;
    a.y += b;
    return a;
}

inline vec3& operator+=(vec3& a, const vec3& b)
{
    a.x += b.x;
    a.y += b.y;
    a.z += b.z;
    return a;
}

inline vec3& operator+=(vec3& a, real b)
{
    a.x += b;
    a.y += b;
    a.z += b;
    return a;
}

inline vec2 operator-(const vec2& a, const vec2& b)
{
    return vec2(a.x - b.x, a.y - b.y);
}

inline vec3 operator-(const vec3& a, const vec3& b)
{
    return vec3(a.x - b.x, a.y - b.y, a.z - b.z);
}

inline vec2 operator-(const vec2& a, real b)
{
    return make_vec2(a.x - b, a.y - b);
}

inline vec2 operator-(real a, const vec2& b)
{
    return make_vec2(a - b.x, a - b.y);
}

inline vec3 operator-(const vec3& a, real b)
{
    return make_vec3(a.x - b, a.y - b, a.z - b);
}

inline vec3 operator-(real a, const vec3& b)
{
    return make_vec3(a - b.x, a - b.y, a - b.z);
}

inline vec2& operator-=(vec2& a, const vec2& b)
{
    a.x -= b.x;
    a.y -= b.y;
    return a;
}

inline vec2& operator-=(vec2& a, real b)
{
    a.x -= b;
    a.y -= b;
    return a;
}

inline vec3& operator-=(vec3& a, const vec3& b)
{
    a.x -= b.x;
    a.y -= b.y;
    a.z -= b.z;
    return a;
}

inline vec3& operator-=(vec3& a, real b)
{
    a.x -= b;
    a.y -= b;
    a.z -= b;
    return a;
}


inline mat2 operator -(const mat2& m)
{
    return make_mat2(
        -m.m00, -m.m01,
        -m.m10, -m.m11);
};

inline mat2 operator+(const mat2& a, const mat2& b)
{
    return make_mat2(
        a.m00 + b.m00, a.m01 + b.m01,
        a.m10 + b.m10, a.m11 + b.m11
    );
};

inline mat2 operator-(const mat2& a, const mat2& b)
{
    return make_mat2(
        a.m00 - b.m00, a.m01 - b.m01,
        a.m10 - b.m10, a.m11 - b.m11);

};

inline mat2& operator+=(mat2& a, const mat2& b)
{
    a.m00 += b.m00;
    a.m01 += b.m01;
    a.m10 += b.m10;
    a.m11 += b.m11;
    return a;
};

inline mat2& operator-=(mat2& a, const mat2& b)
{
    a.m00 -= b.m00;
    a.m01 -= b.m01;
    a.m10 -= b.m10;
    a.m11 -= b.m11;
    return a;
};




// -----------------------------------------------------------------------------
// scalar * vector, vector * scalar, and their compound assignment variants
// -----------------------------------------------------------------------------

inline vec2 operator*(const vec2& a, real b)
{
    return make_vec2(a.x * b, a.y * b);
}

inline vec2 operator*(real a, const vec2& b)
{
    return make_vec2(a * b.x, a * b.y);
}

inline vec3 operator*(const vec3& a, real b)
{
    return make_vec3(a.x * b, a.y * b, a.z * b);
}

inline vec3 operator*(real a, const vec3& b)
{
    return make_vec3(a * b.x, a * b.y, a * b.z);
}

inline mat2 operator*(const mat2& m, real b)
{
    return make_mat2(
        m.m00 * b, m.m01 * b,
        m.m10 * b, m.m11 * b
    );
};

inline mat2 operator*(real a, const mat2& m)
{
    return make_mat2(
        m.m00 * a, m.m01 * a,
        m.m10 * a, m.m11 * a
    );
};

inline vec2& operator*=(vec2& a, real b)
{
    a.x *= b;
    a.y *= b;
    return a;
};

inline vec3& operator*=(vec3& a, real b)
{
    a.x *= b;
    a.y *= b;
    a.z *= b;
    return a;
};

inline mat2& operator*=(mat2& a, real b)
{
    a.m00 *= b;
    a.m01 *= b;
    a.m10 *= b;
    a.m11 *= b;
    return a;
};

// -----------------------------------------------------------------------------
// vector / scalar and matrix / scalar division, and their compound assignment variants


inline vec2 operator/(const vec2& a, real b)
{
    real b_inv = R(1.0) / b;
    return make_vec2(a.x * b_inv, a.y * b_inv);
};

inline vec3 operator/(const vec3& a, real b)
{
    real b_inv = R(1.0) / b;
    return make_vec3(a.x * b_inv, a.y * b_inv, a.z * b_inv);
};

inline mat2 operator/(const mat2& m, real b)
{
    return make_mat2(
        m.m00 / b, m.m01 / b,
        m.m10 / b, m.m11 / b
    );
};


inline vec2& operator/=(vec2& a, real b)
{
    a.x /= b;
    a.y /= b;
    return a;
}

inline vec3& operator/=(vec3& a, real b)
{
    a.x /= b;
    a.y /= b;
    a.z /= b;
    return a;
};

inline mat2& operator/=(mat2& a, real b)
{
    a.m00 /= b;
    a.m01 /= b;
    a.m10 /= b;
    a.m11 /= b;
    return a;
};

// -----------------------------------------------------------------------------
// vector math
// -----------------------------------------------------------------------------

inline real dot(const vec2& a, const vec2& b)
{
    return a.x * b.x + a.y * b.y;
}
inline vec2 dot(const mat2& m, const vec2& v)
{
    return make_vec2(
        m.m00 * v.x + m.m01 * v.y,
        m.m10 * v.x + m.m11 * v.y
    );
};

inline real cross(const vec2& a, const vec2& b)
{
    return a.x * b.y - a.y * b.x;
}

inline vec2 perp(const vec2& v)
{
    return vec2(-v.y, v.x);
}

inline real dot(const vec3& a, const vec3& b)
{
    return a.x * b.x + a.y * b.y + a.z * b.z;
}

inline real length(const vec2& v)
{
    return std::sqrt(v.x * v.x + v.y * v.y);
}

inline real length(const vec3& v)
{
    return std::sqrt(v.x * v.x + v.y * v.y + v.z * v.z);
}

inline vec2 normalize(const vec2& v)
{
    real len = length(v);
    return v / len;
}

inline vec3 normalize(const vec3& v)
{
    real len = length(v);
    return v / len;
}

inline void eigenvalues_symm(const mat2& m, vec2& lambda)
{
    real a = R(0.5) * (m.m00 + m.m11);
    real b = R(0.5) * (m.m00 - m.m11);
    real c = m.m01;

    real r = std::hypot(b, c);

    lambda.x = a + r;
    lambda.y = a - r;
}

inline void spectral_data_symm(const mat2& m, real& trace, real& det, vec2& lambda)
{
    trace = m.m00 + m.m11;
    det = m.m00 * m.m11 - m.m01 * m.m01;

    real a = R(0.5) * trace;
    real b = R(0.5) * (m.m00 - m.m11);
    real c = m.m01;

    real r = std::hypot(b, c);

    lambda.x = a + r;
    lambda.y = a - r;
};

inline real trace(const mat2& m)
{
    return m.m00 + m.m11;
};

inline real det(const mat2& m)
{
    return m.m00 * m.m11 - m.m01 * m.m10;
};

inline mat2 cofactor(const mat2& m)
{
    return make_mat2(
        m.m11, -m.m10,
        -m.m01, m.m00
    );
};

inline mat2 adjugate(const mat2& m)
{
    return make_mat2(
        m.m11, -m.m01,
        -m.m10, m.m00
    );
};

inline mat2 inverse_symm(const mat2& m)
{
    real det = m.m00 * m.m11 - m.m01 * m.m01;
    real inv_det = R(1.0) / det;

    return make_mat2(
         m.m11 * inv_det, -m.m01 * inv_det,
        -m.m01 * inv_det,  m.m00 * inv_det
    );
}

inline mat2 inverse_sqrt_symm(const mat2& m)
{
    real det = m.m00 * m.m11 - m.m01 * m.m01;
    det = std::max(det, R(1e-12));

    real s = std::sqrt(det);
    real tr = m.m00 + m.m11;

    real norm = std::sqrt(std::max(tr + R(2.0) * s, R(1e-12)));

    // invsqrt(m) = sqrt(tr + 2 sqrt(det)) * inv(m + sqrt(det) I)
    mat2 q = make_mat2(
        m.m00 + s, m.m01,
        m.m01,     m.m11 + s
    );

    mat2 inv_q = inverse_symm(q);

    return norm * inv_q;
};

inline mat2 matmul(const mat2& a, const mat2& b)
{
    return make_mat2(
        a.m00 * b.m00 + a.m01 * b.m10, a.m00 * b.m01 + a.m01 * b.m11,
        a.m10 * b.m00 + a.m11 * b.m10, a.m10 * b.m01 + a.m11 * b.m11
    );
};

inline mat2 symmetrize(const mat2& m)
{
    real sym_m01 = R(0.5) * (m.m01 + m.m10);
    return make_mat2(
        m.m00, sym_m01,
        sym_m01, m.m11
    );
};

inline void symmetrize(mat2& m)
{
    real sym_m01 = R(0.5) * (m.m01 + m.m10);
    m.m01 = sym_m01;
    m.m10 = sym_m01;
};

inline mat2 outer(const vec2& a, const vec2& b)
{
    return make_mat2(
        a.x * b.x, a.x * b.y,
        a.y * b.x, a.y * b.y
    );
};

inline mat2 rotate(const mat2& m, real angle)
{
    real c = std::cos(angle);
    real s = std::sin(angle);

    mat2 R = make_mat2(
        c, -s,
        s,  c
    );

    return matmul(R, matmul(m, make_mat2(c, s, -s, c)));
};

inline void rotate(mat2& m, real angle)
{
    real c = std::cos(angle);
    real s = std::sin(angle);

    mat2 R = make_mat2(
        c, -s,
        s,  c
    );

    m = matmul(R, matmul(m, make_mat2(c, s, -s, c)));
};



// -----------------------------------------------------------------------------
// Gauss-Legendre quadrature on [-1, 1]
// -----------------------------------------------------------------------------

struct GaussLegendreQuadrature
{
    int n = 0;
    real* x = nullptr;
    real* w = nullptr;

    GaussLegendreQuadrature() = default;

    explicit GaussLegendreQuadrature(int n_)
    {
        n = n_;
        x = static_cast<real*>(std::malloc(n * sizeof(real)));
        w = static_cast<real*>(std::malloc(n * sizeof(real)));

        if (x == nullptr || w == nullptr)
        {
            std::free(x);
            std::free(w);
            x = nullptr;
            w = nullptr;
            n = 0;
            throw std::bad_alloc();
        }
    }

    ~GaussLegendreQuadrature()
    {
        std::free(x);
        std::free(w);
    }

    GaussLegendreQuadrature(const GaussLegendreQuadrature&) = delete;
    GaussLegendreQuadrature& operator=(const GaussLegendreQuadrature&) = delete;

    GaussLegendreQuadrature(GaussLegendreQuadrature&& other) noexcept
    {
        n = other.n;
        x = other.x;
        w = other.w;

        other.n = 0;
        other.x = nullptr;
        other.w = nullptr;
    }

    GaussLegendreQuadrature& operator=(GaussLegendreQuadrature&& other) noexcept
    {
        if (this != &other)
        {
            std::free(x);
            std::free(w);

            n = other.n;
            x = other.x;
            w = other.w;

            other.n = 0;
            other.x = nullptr;
            other.w = nullptr;
        }

        return *this;
    }
};

inline GaussLegendreQuadrature create_gauss_legendre(
    int n,
    real eps = std::numeric_limits<real>::epsilon() * R(100.0)
)
{
    if (n <= 0)
    {
        throw std::invalid_argument("gauss_legendre: n must be positive");
    }

    GaussLegendreQuadrature gl(n);

    const real pi = std::acos(R(-1.0));
    const int m = (n + 1) / 2;

    for (int i = 0; i < m; ++i)
    {
        real z = std::cos(
            pi * (static_cast<real>(i) + R(0.75)) /
            (static_cast<real>(n) + R(0.5))
        );

        real z_old = z;

        for (int iter = 0; iter < 100; ++iter)
        {
            real P0 = R(1.0);
            real P1 = z;

            if (n > 1)
            {
                for (int k = 2; k <= n; ++k)
                {
                    const real rk = static_cast<real>(k);

                    const real Pk =
                        ((R(2.0) * rk - R(1.0)) * z * P1
                        - (rk - R(1.0)) * P0) / rk;

                    P0 = P1;
                    P1 = Pk;
                }
            }

            const real Pn = P1;
            const real Pnm1 = (n == 1) ? R(1.0) : P0;

            const real dPn =
                static_cast<real>(n) *
                (z * Pn - Pnm1) /
                (z * z - R(1.0));

            z_old = z;
            z = z_old - Pn / dPn;

            if (std::abs(z - z_old) <= eps)
            {
                break;
            }
        }

        real P0 = R(1.0);
        real P1 = z;

        if (n > 1)
        {
            for (int k = 2; k <= n; ++k)
            {
                const real rk = static_cast<real>(k);

                const real Pk =
                    ((R(2.0) * rk - R(1.0)) * z * P1
                    - (rk - R(1.0)) * P0) / rk;

                P0 = P1;
                P1 = Pk;
            }
        }

        const real Pn = P1;
        const real Pnm1 = (n == 1) ? R(1.0) : P0;

        const real dPn =
            static_cast<real>(n) *
            (z * Pn - Pnm1) /
            (z * z - R(1.0));

        const real weight =
            R(2.0) / ((R(1.0) - z * z) * dPn * dPn);

        gl.x[i] = -z;
        gl.x[n - 1 - i] = z;

        gl.w[i] = weight;
        gl.w[n - 1 - i] = weight;
    }

    return gl;
}

// Map Gauss-Legendre point/weight from [-1, 1] to [a, b].
inline real gauss_legendre_point(real xi, real a, real b)
{
    return R(0.5) * ((b - a) * xi + (b + a));
}

inline real gauss_legendre_weight(real wi, real a, real b)
{
    return R(0.5) * (b - a) * wi;
}
