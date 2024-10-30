#pragma once

#include <math.h>
#include <cstdlib>

namespace bb {
    using Real = double;
    using RActivation = Real (*) (Real);
    // TODO: Deriv of activation func.
    
    /**
     * @brief Absolute valid of Real.
     * 
     * @param a Input.
     * @return |Input|
     */
    inline Real rabs(Real a) {
        return fabs(a);
    }

    inline Real rsqrt(Real a) {
        return static_cast<Real>(sqrtl(static_cast<long double>(a)));
    }

    /**
     * @brief Random Real within a range.
     * 
     * This function doesn't check for a valid range for performance.
     * 
     * @param min Minimum valid allowed.
     * @param max Maximum valid allowed.
     * @return Random value within range.
     */
    inline Real rrand(Real min, Real max) {
        return static_cast<Real>(rand()) / static_cast<Real>(RAND_MAX) * (max - min) + min;
    }

    /**
     * @brief Determine if two values are the same, given a tolerance.
     * 
     * @param a Value a.
     * @param b Value b.
     * @param tolerance Tolerance in which they will still be considered "equal".
     * @return true If a and b are effectively equal.
     * @return false If a and b are not equal.
     */
    inline bool repsilon(Real a, Real b, Real tolerance) {
        return rabs(a - b) < tolerance;
    }

    /**
     * @brief Determine if two values are the same, given a tolerance.
     * 
     * @param a Value a.
     * @param b Value b.
     * @return true If a and b are effectively equal.
     * @return false If a and b are not equal.
     */
    inline bool repsilon(Real a, Real b) {
        return repsilon(a, b, 0.001);
    }
    
}

typedef unsigned short ushort;
typedef unsigned int uint;