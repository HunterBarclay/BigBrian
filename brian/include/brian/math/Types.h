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

    /**
     * @brief Random Real within a range.
     * 
     * This function doesn't check for a valid range for performance.
     * 
     * @param p_min Minimum valid allowed.
     * @param p_max Maximum valid allowed.
     * @return Random value within range.
     */
    inline Real rrand(Real p_min, Real p_max) {
        return static_cast<Real>(rand()) / static_cast<Real>(RAND_MAX) * (p_max - p_min) + p_min;
    }

    /**
     * @brief Determine if two values are the same, given a tolerance.
     * 
     * @param p_a Value a.
     * @param p_b Value b.
     * @param p_tolerance Tolerance in which they will still be considered "equal".
     * @return true If a and b are effectively equal.
     * @return false If a and b are not equal.
     */
    inline bool repsilon(Real p_a, Real p_b, Real p_tolerance) {
        return rabs(p_a - p_b) < p_tolerance;
    }

    /**
     * @brief Determine if two values are the same, given a tolerance.
     * 
     * @param p_a Value a.
     * @param p_b Value b.
     * @return true If a and b are effectively equal.
     * @return false If a and b are not equal.
     */
    inline bool repsilon(Real p_a, Real p_b) {
        return repsilon(p_a, p_b, 0.001);
    }
    
}

typedef unsigned short ushort;
typedef unsigned int uint;